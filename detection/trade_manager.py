"""
Trade Manager v1.0 — Real-money position management for SMC signals.

Responsibilities:
  - Listen for signals from SMC scanner (LONG/SHORT)
  - Open positions on Bybit when conditions allow
  - Monitor open positions every 10s
  - Close/partial-close based on configurable exit rules
  - Send Telegram notifications on every action
  - Persist positions and recent trades in DB

Position sizing (3 modes):
  - 'fixed_usd'    : qty = $usd_amount / entry_price          (default $100)
  - 'fixed_pct'    : qty = (balance * pct/100) / entry_price
  - 'risk_based'   : qty such that loss at SL = balance * 1%

Exit rules (all toggleable, evaluated per priority):
  1. Stop Loss          (fixed % or absolute)
  2. Take Profit        (fixed % or absolute)
  3. Reverse SMC signal (CHoCH only, A1)
  4. HTF trend flip     (when HTF filter active)
  5. Time stop          (close if open longer than X hours)
  6. Trailing Stop      (track high water, trail by Y%)
  7. Break-Even Move    (move SL to entry after +X%)
  8. BOS-N partial close (close X% at 2nd/3rd/4th BOS, then trailing)

This module is STATEFUL and SAFETY-CRITICAL. Default toggle = OFF.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ======== DB keys ========
DB_KEY_TM_SETTINGS = 'tm_settings'
DB_KEY_TM_POSITIONS = 'tm_positions'
DB_KEY_TM_CLOSED = 'tm_closed_trades'
DB_KEY_TM_SHADOW = 'tm_shadow_positions'   # paper-trading positions
DB_KEY_TM_SHADOW_CLOSED = 'tm_shadow_closed'

MONITOR_INTERVAL_SECS = 10
CLOSED_TRADES_LIMIT = 2_000   # IN-MEMORY / UI rolling cap only. EVERY closed
                            # trade is still kept FOREVER in the permanent DB
                            # archive (TradeArchive, archive_trades=True) — «всі
                            # зберігаємо». This cap just stops the in-RAM blob
                            # (each trade carries its chronology) from growing
                            # unbounded and OOM-killing the 512MB Render instance.
                            # Raise only with more RAM; full history lives in DB.
INITIAL_DELAY_SECS = 20      # wait at startup before first tick
TRADE_LOG_MAX = 5000         # max per-trade time-series samples kept (safety cap)


def _lite_trade(d: Dict) -> Dict:
    """Copy a position/closed-trade dict WITHOUT the heavy per-trade history
    time-series (kept out of the /api/tm/state payload). Exposes only
    `hist_len` so the UI knows a chart is available; the full series is
    fetched on demand via /api/tm/trade-history."""
    h = d.get('history') or []
    out = {k: v for k, v in d.items() if k not in ('history', '_last_log_at')}
    out['hist_len'] = len(h)
    return out

# Reconcile interval is now user-tunable via DEFAULT_SETTINGS["reconcile_interval_secs"].
# At each monitor tick we read the setting and compute N_TICKS dynamically,
# so users can change it without restart. Min 5s (rounded to 10s), Max 300s.

DEFAULT_SETTINGS = {
    # Master toggle — DEFAULT OFF for safety
    'enabled': False,
    
    # === Position Sizing ===
    'sizing_mode': 'fixed_usd',     # 'fixed_usd' | 'fixed_pct' | 'risk_based'
    'fixed_usd_amount': 100.0,       # USD per trade (mode = fixed_usd)
    'fixed_pct_balance': 2.0,        # % of balance (mode = fixed_pct)
    'risk_pct_balance': 1.0,         # max loss as % of balance (mode = risk_based)
    'leverage': 10,                  # 1-50 typically
    'max_open_positions': 10,        # max REAL positions; if exceeded → Test Mode (if on)

    # === Exit Rules — each toggleable ===
    'use_sl': True,                  # Stop Loss
    'sl_pct': 2.0,                   # %
    
    # === SL strategy ===
    # Two modes determine WHERE the stop is placed when use_sl is on:
    #   'pct'           — classic % offset from entry price (sl_pct above).
    #   'volumized_ob'  — anchor SL to the Volumized OB block boundary:
    #                       LONG  → ob_bottom × (1 − sl_vob_buffer_pct/100)
    #                       SHORT → ob_top    × (1 + sl_vob_buffer_pct/100)
    #                     Available only when the SMC scanner has the
    #                     Volumized OB Trend filter ENABLED (use_volumized_ob)
    #                     — without that filter we don't maintain the OB-meta
    #                     cache so there's nothing to anchor against. Falls
    #                     back to 'pct' on any failure (filter off, missing
    #                     cached OB meta, OB direction mismatch, or computed
    #                     SL on the wrong side of entry).
    'sl_mode': 'pct',
    'sl_vob_buffer_pct': 0.2,        # % buffer beyond OB boundary
    
    'use_tp': True,                  # Take Profit
    'tp_pct': 5.0,                   # %

    # ── 🤖 Bot-side SOFT exit layer (independent of the exchange-native
    # SL/TP/Trailing above). The exchange orders stay as a hard backstop; this
    # layer has the BOT itself watch live PnL% per position and close on its own
    # rules — identically for REAL and PAPER (virtual) trades. ──
    'bot_exit_enabled': False,       # master toggle for the whole soft layer
    'bot_sl_enabled': False,         # soft Stop Loss (bot-monitored, fixed %)
    'bot_sl_pct': 0.8,               # close when pnl% <= -this
    'bot_tp_enabled': False,         # soft trailing Take Profit (ladder)
    'bot_tp_pct': 1.5,              # activation: start trailing once peak >= this
    'bot_tp_giveback_pct': 0.5,      # close when pnl retraces this % below peak

    'use_reverse_smc': True,         # Close on opposite CHoCH
    'use_reverse_signal': False,     # Flip position on opposite QUALIFIED signal (off by default)

    # === 🔄 CTR reversal-after-peak exit ===
    # Close a position once it TOPPED OUT and a reversal is confirmed by the
    # CTR + supporting readings (the same series shown on the trade chart):
    #   1. peak PnL reached ≥ ctr_reversal_min_peak_pct (there WAS a peak);
    #   2. CTR lean turned to the OPPOSITE extreme ≥ ctr_reversal_ctr_pct
    #      (LONG → SHORT-нахил, SHORT → LONG-нахил);
    #   3. ≥ ctr_reversal_confirmations of 3 confirm a real decline:
    #        (a) PnL gave back ≥ ctr_reversal_giveback_pct % of the peak,
    #        (b) exhaustion ≥ ctr_reversal_exh_pct,
    #        (c) ММ (fuel) turned against / went weak.
    # Fires on real AND paper. The open-positions table shows a live readiness %.
    'use_ctr_reversal_exit': True,
    'ctr_reversal_min_peak_pct': 1.0,
    'ctr_reversal_ctr_pct': 70,
    'ctr_reversal_giveback_pct': 30,
    'ctr_reversal_exh_pct': 66,
    'ctr_reversal_confirmations': 2,
    # Profit-lock floor: the reversal exit fires ONLY while current PnL ≥ this %
    # (default 0 = never close at a loss — that's the SL's job, not this rule).
    'ctr_reversal_min_lock_pct': 0.0,
    # Persistence guard (in BARS of the CTR timeframe): the reversal condition
    # must hold continuously for ≥ this many closed bars before the trade is
    # closed — so a tiny intrabar wobble can't exit and give back the potential.
    # 0 = react instantly (old behaviour). 2 ≈ two CTR bars of confirmation.
    'ctr_reversal_min_bars': 2,

    'use_htf_flip': True,            # Close when HTF trend flips against us
    
    'use_time_stop': False,          # Close after N hours regardless
    'time_stop_hours': 4,
    
    'use_trailing': True,            # Trailing stop
    'trailing_activate_pct': 1.0,    # activate after +X% profit
    'trailing_distance_pct': 0.5,    # trail by Y% from peak
    
    'use_be': True,                  # Break-Even Move
    'be_trigger_pct': 0.5,           # move SL to entry after +X% profit
    # === BE commission buffer ===
    # When BE fires, SL is normally placed exactly at entry. If the SL
    # then triggers, the trade closes at ~0 PnL but trading FEES (maker/
    # taker on Bybit, applied to both open and close orders) push the net
    # PnL slightly negative. Adding a small offset past entry compensates:
    #     LONG  SL → entry × (1 + buf/100)   ← slightly ABOVE entry
    #     SHORT SL → entry × (1 − buf/100)   ← slightly BELOW entry
    # Default 0.12% ≈ round-trip taker fee on Bybit Perpetual (0.06% × 2).
    # Range 0..1%. Set to 0 to disable buffer (legacy behavior).
    'be_commission_buffer_pct': 0.12,
    
    # === Forecast 1H Confluence Close exit ===
    # Closes position when both:
    #   1. Opposite CHoCH appears on LTF (after position open)
    #   2. Forecast 1H side is opposite to position
    # Either condition alone won't close — needs both (confluence).
    'use_forecast_1h_close': True,
    
    # === Opposite-OB exit (configurable timeframe) ===
    # Closes position when an OB with the OPPOSITE bias appears on the
    # configured timeframe. This is a faster mean-reversion exit than
    # HTF flip or Forecast — it triggers as soon as the LTF structure
    # produces a counter-direction OB, regardless of CHoCH/BOS tag.
    # Example: position LONG, then on the configured TF an OB with
    #          bias=BEARISH appears → close immediately.
    # Default OFF + 15m TF — opt-in for users who want the strict-bias-flip exit.
    'use_opposite_ob_exit': False,
    'opposite_ob_exit_timeframe': '15m',  # 15m / 30m / 1h / 4h
    
    # === Test mode (paper trading for exit-rule validation) ===
    # When ON: signals create "shadow" positions tracked in memory only.
    # No Bybit orders. Exit rules still evaluate and send Telegram alerts
    # so the user can validate strategy behavior without real risk.
    # Real positions (when TM enabled=True) take precedence over shadow.
    'test_mode': True,
    
    # === Global directional gates ===
    # Master toggles that gate ALL position opens — real and shadow alike —
    # regardless of signal source (SMC, manual UI, future webhooks).
    # Closures/exits/reverses still run normally; only the OPEN step is
    # blocked. Both default ON so behavior is unchanged after upgrade.
    # The UI exposes these as two prominent buttons at the top of Smart
    # Money so the operator can flip them quickly without scrolling
    # through settings.
    'allow_long_entries': True,
    'allow_short_entries': True,

    # === Per-trade time-series log ===
    # When ON, every open position (real + test) is sampled every
    # trade_log_interval_sec (price / PnL% / ММ / exhaustion). The series is
    # carried into the closed-trade record so its full history can be charted
    # by clicking the closed trade.
    'trade_log_enabled': False,
    # 20s (not 60s) → denser samples so the trade chart pinpoints the turn (CTR
    # crossover reset, price peak, profit↔loss flip) much more precisely. Monitor
    # cadence is 10s, so 20s is safe; TRADE_LOG_MAX (5000) still covers ~27h.
    'trade_log_interval_sec': 20,
    # Decision Center TRAJECTORY sampling cadence (heavier than a price sample →
    # coarser). Captured into the trade history so we can replay how the verdict
    # evolved. Min 60s enforced in code.
    'trade_log_decision_interval_sec': 180,

    # === Fuel Auto-Filter confirmation gate ===
    # When ON, a trade is only opened if the SAME coin with the SAME direction
    # is currently present in the ❤️ Fuel Auto-Filter table (i.e. its fuel has
    # held for the configured threshold). If the coin+direction is NOT in the
    # FF table, the open is IGNORED. This makes FF a mandatory confirmation
    # for every entry. Default ON.
    'require_fuel_confirm': True,

    # === Telegram notification toggles ===
    # Independently control Telegram alerts for real positions vs paper trades.
    # SMC scanner always sends its own signal alerts — these toggles control
    # the EXTRA alerts from Trade Manager about position lifecycle.
    'telegram_alerts': True,        # real position open/close/partial
    'test_telegram_alerts': True,   # paper [TEST] open/close
    
    # === Bybit Position Reconciliation Interval ===
    # How often (in seconds) to sync TM's view with actual Bybit positions:
    #   - Adopt positions opened manually on Bybit / leftover from prior runs
    #   - Detect external closes (manual close, Bybit-side SL/TP, liquidation)
    # Rounded internally to the nearest multiple of monitor tick (10s).
    # Recommended: 10-15s. Min 5s (rounded up to 10s), Max 300s.
    # Bybit rate limit on get_positions is 5-10 req/s on UID; at 10s interval
    # we use ~0.1 req/s, very safe.
    'reconcile_interval_secs': 10,

    # === Trade archive (ML training dataset) ===
    # When ON, every closed trade is also written to the long-term trade
    # archive (for ML training / analytics), in addition to the capped
    # Recent Closed list. Read in _finalize/close via settings['archive_trades'].
    # MUST live here so update_settings() whitelists it and it persists —
    # otherwise the /api/trade-archive/toggle write is silently dropped.
    # Default ON: keeps EVERY closed trade permanently in the DB archive so the
    # in-memory rolling list (CLOSED_TRADES_LIMIT) can stay small without losing
    # history — «всі зберігаємо», але без роздування RAM.
    'archive_trades': True,

    # === BOS-N partial closes (after CHoCH+BOS opening) ===
    # The opening trade counts the entry-BOS as #1.
    # Subsequent same-direction BOS events are #2, #3, #4...
    'use_bos_partials': True,
    'bos_2_close_pct': 70,           # close 70% on BOS-2
    'bos_3_close_pct': 0,            # close additional 0% on BOS-3 (off by default)
    'bos_4_close_pct': 0,
    'trailing_after_bos_2': True,    # auto-activate trailing after BOS-2 partial
    
    # === Position Health Score (advisory rule-based AI) ===
    # An expert system that aggregates HTF / Forecast 1H / CTR / LTF
    # structure / PnL momentum / time-decay into a single -100..+100 score.
    # The score is informational ONLY in this iteration — it is shown in the
    # UI and Telegram, but does NOT close positions automatically. Users
    # who want auto-close based on the score can wire that in later by
    # promoting `health_score_action` from 'advise' to 'close'.
    'health_score_enabled': True,
    'health_score_preset': 'balanced',     # 'aggressive' | 'balanced' | 'conservative'
    # Per-component weight overrides. Empty dict = use evaluator defaults.
    # Slider UI in Smart Money page populates these. Each value is a positive
    # number representing the maximum points that component can contribute.
    'health_score_weights': {
        'weight_htf_alignment': 25.0,
        'weight_forecast_1h': 14.0,
        'weight_forecast_4h': 18.0,
        'weight_ltf_choch': 25.0,
        'weight_ltf_bos': 12.0,
        'weight_ctr_alignment': 15.0,
        'weight_ctr_zone': 10.0,
        'weight_pnl_momentum': 10.0,
        'weight_time_decay': 10.0,
    },
    # If non-null, overrides the threshold from the preset. Used when the
    # user fine-tunes from the UI without picking a different preset.
    'health_score_threshold_override': None,
    
    # === Entry Score (advisory rule-based AI for position OPENING) ===
    # Mirror image of Health Score but for the "should we open this?"
    # decision. Score is in [-100, +100] like exit, but threshold is
    # POSITIVE (we want a signal good enough to act on, e.g. +30).
    # Like exit, this is ADVISORY in this iteration: shown in Telegram
    # OPEN messages and stored on the position so UI can show the snapshot.
    # Does NOT block opens — user explicitly chose advisory mode.
    'entry_score_enabled': True,
    'entry_score_preset': 'balanced',
    'entry_score_weights': {
        'weight_htf_alignment': 25.0,
        'weight_forecast_1h': 12.0,
        'weight_forecast_4h': 15.0,
        'weight_ctr_alignment': 15.0,
        'weight_ctr_zone': 10.0,
        'weight_choch_freshness': 12.0,
        'weight_pivot_proximity': 12.0,
        'weight_pd_zone': 15.0,
        'weight_volume_confirmation': 8.0,
        'weight_atr_health': 8.0,
    },
    'entry_score_threshold_override': None,
}


# Per-position state (in memory + persisted as dict)
def _new_position(symbol, side, entry_price, qty, sl_price, tp_price, order_id, opened_by):
    return {
        'symbol': symbol,
        'side': side,                    # 'LONG' | 'SHORT'
        'entry_price': float(entry_price),
        'qty': float(qty),
        'remaining_qty': float(qty),     # decreases on partial closes
        'sl_price': float(sl_price) if sl_price else None,
        'tp_price': float(tp_price) if tp_price else None,
        'opened_at': time.time(),
        'opened_by': opened_by,          # 'choch_bos' | 'choch'
        'order_id': order_id,
        # Tracking
        'be_moved': False,
        'trailing_active': False,
        'trailing_peak': float(entry_price),  # high-water mark for LONG; low for SHORT
        # Marker — set by on_bos_event when BOS-2 hook activates trailing.
        # _update_trailing checks this so that BOS-2 trailing works even
        # when the global use_trailing toggle is off. Default False means
        # legacy positions loaded before this field existed continue to
        # behave as if trailing is purely global-toggle controlled.
        'trailing_via_bos2': False,
        'partial_closes_done': [],       # ['bos_2', 'bos_3', ...]
        'bos_count_since_entry': 1,      # the opening BOS itself counts as #1
    }


class TradeManager:
    
    def __init__(self, db=None, notifier=None, bybit=None, scanner=None):
        self.db = db
        self.notifier = notifier
        self.bybit = bybit
        self.scanner = scanner   # SMC scanner reference (for tradeable list, HTF state)
        
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # In-memory state
        self._settings = self._load_settings()
        self._positions: Dict[str, Dict] = {}     # symbol → position dict
        self._closed_trades: List[Dict] = []

        # Live exchange-truth snapshot of open positions, keyed by symbol.
        # Populated from Bybit on every reconcile tick (and refreshed on-demand
        # by get_state when stale). Holds the EXACT exchange values — avgPrice,
        # markPrice, unrealisedPnl, size, leverage, liqPrice — so the UI shows
        # reality instead of our own approximation. {symbol: {...}} + timestamp.
        self._live_positions: Dict[str, Dict] = {}
        self._live_positions_at: float = 0.0
        # Throttle for on-demand refresh from get_state so frequent UI polls
        # don't hammer the Bybit API. Refresh at most once per this interval.
        self._live_refresh_min_interval: float = 3.0

        # Shadow (paper) positions — tracked when test_mode=True even if TM is
        # disabled. Used to validate exit rules without real Bybit orders.
        self._shadow_positions: Dict[str, Dict] = {}
        self._shadow_closed: List[Dict] = []
        
        # Per-position state for the Health Score evaluator. Keyed by symbol,
        # parallel to _positions / _shadow_positions. Tracks runtime stats
        # that are NOT persisted in the position dict itself (so we don't
        # bloat the canonical position record):
        #   peak_pnl_pct: highest PnL the position has reached
        #   bos_with_count / bos_against_count: BOS events since open
        #   last_choch_after_open: latest CHoCH event observed AFTER open
        # When a position closes, its entry is removed from these dicts.
        self._pos_state: Dict[str, Dict] = {}
        self._shadow_pos_state: Dict[str, Dict] = {}
        # SMC Hold-Confidence cache: {symbol: (ts, result)}. Recomputed at
        # most once per HOLD_SCORE_TTL to keep /api/tm/state (polled ~5s)
        # cheap, since each score runs a full SMC analysis on klines.
        self._hold_cache: Dict[str, tuple] = {}
        
        self._load_positions()
        self._load_closed_trades()
        self._load_shadow_positions()
        self._load_shadow_closed()
        
        # Initialize evaluator state for any positions we restored from DB.
        # CRITICAL: seed the peak_pnl_pct from what the trade already reached
        # BEFORE the restart — from the position's own persisted field and its
        # recorded history samples. Otherwise a restart (Render sleep/wake) would
        # reset the peak to 0, and the 🤖 soft trailing-TP would fail to
        # re-activate for a trade that had already peaked (it would then fall all
        # the way to the soft SL). This keeps the live trailing consistent with
        # the chart across restarts.
        for sym, pos in self._positions.items():
            st = self._fresh_pos_state()
            st['peak_pnl_pct'] = self._history_peak(pos)
            self._pos_state[sym] = st
        for sym, pos in self._shadow_positions.items():
            st = self._fresh_pos_state()
            st['peak_pnl_pct'] = self._history_peak(pos)
            self._shadow_pos_state[sym] = st
        
        # Stats
        self._tick_count = 0
        self._errors = 0
    
    # ============================================================
    # Persistence
    # ============================================================
    
    def _load_settings(self) -> Dict:
        if not self.db:
            return DEFAULT_SETTINGS.copy()
        try:
            stored = self.db.get_setting(DB_KEY_TM_SETTINGS, None)
            if isinstance(stored, dict):
                merged = DEFAULT_SETTINGS.copy()
                merged.update(stored)
                return merged
        except:
            pass
        return DEFAULT_SETTINGS.copy()
    
    def _persist_settings(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TM_SETTINGS, self._settings)
            except Exception as e:
                print(f"[TM] Settings persist error: {e}")
    
    def _load_positions(self):
        if not self.db:
            return
        try:
            stored = self.db.get_setting(DB_KEY_TM_POSITIONS, {})
            if isinstance(stored, dict):
                self._positions = stored
                print(f"[TM] Loaded {len(self._positions)} open positions from DB")
        except Exception as e:
            print(f"[TM] Position load error: {e}")
    
    def _persist_positions(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TM_POSITIONS, self._positions)
            except Exception as e:
                print(f"[TM] Position persist error: {e}")
    
    def _load_closed_trades(self):
        if not self.db:
            return
        try:
            stored = self.db.get_setting(DB_KEY_TM_CLOSED, [])
            if isinstance(stored, list):
                self._closed_trades = stored[-CLOSED_TRADES_LIMIT:]
        except:
            pass
    
    def _persist_closed_trades(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TM_CLOSED,
                                     self._closed_trades[-CLOSED_TRADES_LIMIT:])
            except Exception as e:
                print(f"[TM] Closed-trades persist error: {e}")
    
    def export_closed_trades(self) -> Dict:
        """Full trade records (real + paper), CLOSED and currently-OPEN, WITH
        their per-trade history (chronology) — for the 🧾 activity-log analytical
        export, which links each log session to the trade it produced. Open trades
        are included so a freshly-opened position (still running) is linked too,
        not only after it closes. Deep-ish copies under the lock."""
        with self._lock:
            real = [dict(c) for c in self._closed_trades]
            shadow = [dict(c) for c in self._shadow_closed]
            open_real = [dict(p) for p in self._positions.values()]
            open_shadow = [dict(p) for p in self._shadow_positions.values()]
        return {'real': real, 'shadow': shadow,
                'open_real': open_real, 'open_shadow': open_shadow}

    def _load_shadow_positions(self):
        if not self.db:
            return
        try:
            stored = self.db.get_setting(DB_KEY_TM_SHADOW, {})
            if isinstance(stored, dict):
                self._shadow_positions = stored
                if stored:
                    print(f"[TM] Loaded {len(stored)} shadow positions from DB")
        except Exception as e:
            print(f"[TM] Shadow positions load error: {e}")
    
    def _persist_shadow_positions(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TM_SHADOW, self._shadow_positions)
            except Exception as e:
                print(f"[TM] Shadow positions persist error: {e}")
    
    def _load_shadow_closed(self):
        if not self.db:
            return
        try:
            stored = self.db.get_setting(DB_KEY_TM_SHADOW_CLOSED, [])
            if isinstance(stored, list):
                self._shadow_closed = stored[-CLOSED_TRADES_LIMIT:]
        except:
            pass
    
    def _persist_shadow_closed(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TM_SHADOW_CLOSED,
                                     self._shadow_closed[-CLOSED_TRADES_LIMIT:])
            except Exception as e:
                print(f"[TM] Shadow closed persist error: {e}")

    # ============================================================
    # Settings API
    # ============================================================
    
    def get_settings(self) -> Dict:
        with self._lock:
            return dict(self._settings)
    
    def update_settings(self, new: Dict) -> Dict:
        with self._lock:
            allowed = list(DEFAULT_SETTINGS.keys())
            for k in allowed:
                if k in new:
                    self._settings[k] = new[k]
            
            # Type coercion / clamping
            self._settings['enabled'] = bool(self._settings.get('enabled', False))
            
            if self._settings.get('sizing_mode') not in ('fixed_usd', 'fixed_pct', 'risk_based'):
                self._settings['sizing_mode'] = 'fixed_usd'
            
            try:
                self._settings['leverage'] = max(1, min(50, int(self._settings.get('leverage', 10))))
            except:
                self._settings['leverage'] = 10

            try:
                self._settings['max_open_positions'] = max(1, min(50, int(self._settings.get('max_open_positions', 10))))
            except:
                self._settings['max_open_positions'] = 10

            for k in ['fixed_usd_amount', 'fixed_pct_balance', 'risk_pct_balance',
                      'sl_pct', 'tp_pct', 'time_stop_hours',
                      'sl_vob_buffer_pct',
                      'trailing_activate_pct', 'trailing_distance_pct', 'be_trigger_pct',
                      'be_commission_buffer_pct',
                      'bot_sl_pct', 'bot_tp_pct', 'bot_tp_giveback_pct',
                      'bos_2_close_pct', 'bos_3_close_pct', 'bos_4_close_pct']:
                try:
                    self._settings[k] = float(self._settings.get(k, DEFAULT_SETTINGS.get(k, 0)))
                except:
                    self._settings[k] = float(DEFAULT_SETTINGS.get(k, 0))
            
            # sl_mode — validated string. Unknown → 'pct' (safe fallback).
            if self._settings.get('sl_mode') not in ('pct', 'volumized_ob'):
                self._settings['sl_mode'] = 'pct'
            # sl_vob_buffer_pct — clamp to a sane band. 0 = SL exactly at the
            # OB boundary (risky — slightest wick hits). 5 = a generous 5%
            # past the block. Default 0.2 mirrors the user's earlier ATR-buffer
            # convention used by the LuxAlgo PRO Engine SL output.
            self._settings['sl_vob_buffer_pct'] = max(0.0, min(5.0,
                float(self._settings.get('sl_vob_buffer_pct', 0.2))))
            
            # Reconcile interval — user-tunable Bybit sync cadence. Allowed
            # range [5, 300] seconds. Below 5 = wastes API quota for no benefit
            # (monitor tick is 10s anyway). Above 300 = reaction to manual
            # trades feels laggy. Default 10s = same as monitor tick = perfect
            # responsiveness with ~0.1 req/s API load. Coerced to int.
            try:
                ri = int(float(self._settings.get('reconcile_interval_secs', 10)))
            except (TypeError, ValueError):
                ri = 10
            self._settings['reconcile_interval_secs'] = max(5, min(300, ri))
            # be_commission_buffer_pct — clamp to a sane band. 0 means no
            # buffer (legacy: SL exactly at entry). 1% is the upper bound;
            # anything bigger isn't "fee compensation" anymore, it's a
            # profit lock-in (use trailing for that).
            self._settings['be_commission_buffer_pct'] = max(0.0, min(1.0,
                float(self._settings.get('be_commission_buffer_pct', 0.12))))
            
            for k in ['use_sl', 'use_tp', 'use_reverse_smc', 'use_reverse_signal',
                      'use_htf_flip',
                      'use_time_stop', 'use_trailing', 'use_be',
                      'use_forecast_1h_close', 'use_opposite_ob_exit',
                      'test_mode',
                      'allow_long_entries', 'allow_short_entries',
                      'require_fuel_confirm',
                      'telegram_alerts', 'test_telegram_alerts',
                      'use_bos_partials', 'trailing_after_bos_2',
                      'health_score_enabled',
                      'bot_exit_enabled', 'bot_sl_enabled', 'bot_tp_enabled',
                      'entry_score_enabled', 'use_ctr_reversal_exit']:
                self._settings[k] = bool(self._settings.get(k, False))
            
            # opposite_ob_exit_timeframe — validated string, default '15m'
            ALLOWED_EXIT_TFS = ('15m', '30m', '1h', '4h')
            if self._settings.get('opposite_ob_exit_timeframe') not in ALLOWED_EXIT_TFS:
                self._settings['opposite_ob_exit_timeframe'] = '15m'
            
            # === Health Score validation ===
            preset = self._settings.get('health_score_preset', 'balanced')
            if preset not in ('aggressive', 'balanced', 'conservative'):
                self._settings['health_score_preset'] = 'balanced'
            
            # Threshold override: None or float
            tov = self._settings.get('health_score_threshold_override')
            if tov is None or tov == '':
                self._settings['health_score_threshold_override'] = None
            else:
                try:
                    self._settings['health_score_threshold_override'] = float(tov)
                except (TypeError, ValueError):
                    self._settings['health_score_threshold_override'] = None
            
            # Weights: dict of floats. Sanitize each value, falling back to
            # the default weight when the user supplied something invalid.
            weights = self._settings.get('health_score_weights')
            default_weights = DEFAULT_SETTINGS['health_score_weights']
            if not isinstance(weights, dict):
                weights = {}
            cleaned = {}
            for wkey, default_val in default_weights.items():
                try:
                    val = float(weights.get(wkey, default_val))
                    # Clamp to a reasonable range so a UI slider mishap can't
                    # drive scores into nonsense territory
                    cleaned[wkey] = max(0.0, min(100.0, val))
                except (TypeError, ValueError):
                    cleaned[wkey] = float(default_val)
            self._settings['health_score_weights'] = cleaned
            
            # === Entry Score validation (mirror of Health Score) ===
            preset = self._settings.get('entry_score_preset', 'balanced')
            if preset not in ('aggressive', 'balanced', 'conservative'):
                self._settings['entry_score_preset'] = 'balanced'
            
            tov = self._settings.get('entry_score_threshold_override')
            if tov is None or tov == '':
                self._settings['entry_score_threshold_override'] = None
            else:
                try:
                    self._settings['entry_score_threshold_override'] = float(tov)
                except (TypeError, ValueError):
                    self._settings['entry_score_threshold_override'] = None
            
            entry_weights = self._settings.get('entry_score_weights')
            default_entry_weights = DEFAULT_SETTINGS['entry_score_weights']
            if not isinstance(entry_weights, dict):
                entry_weights = {}
            cleaned_e = {}
            for wkey, default_val in default_entry_weights.items():
                try:
                    val = float(entry_weights.get(wkey, default_val))
                    cleaned_e[wkey] = max(0.0, min(100.0, val))
                except (TypeError, ValueError):
                    cleaned_e[wkey] = float(default_val)
            self._settings['entry_score_weights'] = cleaned_e
            
            self._persist_settings()
            
            # If toggle was just turned ON and not running — start
            if self._settings['enabled'] and not self._running:
                self.start()
            elif not self._settings['enabled'] and self._running:
                # Don't stop monitoring (positions still need management)
                # but new signals will be ignored
                pass
            
            return dict(self._settings)
    
    def is_enabled(self) -> bool:
        return self._settings.get('enabled', False)
    
    def _side_allowed(self, side: str) -> bool:
        """Global directional gate. Returns False when the configured
        master toggle for that direction is OFF. Used by both _open_position
        (real) and _open_shadow (paper) to enforce the same rule everywhere.
        Defaults to True for both sides so old deploys without the keys keep
        their current behavior.
        """
        s = self._settings
        if side == 'LONG':
            return bool(s.get('allow_long_entries', True))
        if side == 'SHORT':
            return bool(s.get('allow_short_entries', True))
        return True
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._loop, daemon=True, name="TradeManager")
        self._monitor_thread.start()
        print(f"[TM] ✅ Started: enabled={self.is_enabled()}, "
              f"open_positions={len(self._positions)}")
        # Trigger an initial reconcile on a separate thread so start() returns
        # quickly. The first tick will run reconcile too (via the periodic
        # counter at startup → _tick_count == 1 % RECONCILE_EVERY_N_TICKS
        # won't fire on tick 1; but we don't want to wait), so we kick a
        # one-shot async call here.
        threading.Thread(target=self._reconcile_with_bybit,
                          daemon=True, name="TM-InitReconcile").start()
    
    def stop(self):
        self._running = False
    
    def _loop(self):
        print("[TM] 🧵 Monitor thread started")
        time.sleep(INITIAL_DELAY_SECS)
        while self._running:
            try:
                self._tick()
            except Exception as e:
                self._errors += 1
                if self._errors <= 5:
                    print(f"[TM] Tick error: {e}")
            for _ in range(MONITOR_INTERVAL_SECS):
                if not self._running:
                    return
                time.sleep(1)
    
    def _tick(self):
        """Monitor each open position for exit conditions."""
        self._tick_count += 1
        
        # Periodic reconciliation with Bybit so TM picks up positions opened
        # outside it (manual trades, leftover positions, etc.) and detects
        # external closes (manual close, Bybit-side SL/TP fill, liquidation).
        # Interval is user-tunable via settings. Effective N_TICKS is at
        # least 1 (one tick = 10s minimum due to monitor cadence).
        import math
        interval = self._settings.get('reconcile_interval_secs', 10)
        try:
            interval = float(interval)
            if not math.isfinite(interval):
                interval = 10
        except (TypeError, ValueError):
            interval = 10
        # Clamp to allowed range [5, 300] then round to nearest tick multiple
        interval = max(5, min(300, interval))
        n_ticks = max(1, round(interval / MONITOR_INTERVAL_SECS))
        if self._tick_count % n_ticks == 0:
            try:
                self._reconcile_with_bybit()
            except Exception as e:
                if self._errors <= 10:
                    print(f"[TM] Reconcile error: {e}")
                self._errors += 1
        
        with self._lock:
            symbols = list(self._positions.keys())
            shadow_symbols = list(self._shadow_positions.keys())
        
        for sym in symbols:
            try:
                self._monitor_position(sym)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[TM] Monitor error {sym}: {e}")
                self._errors += 1
        
        # Shadow positions: only manual SL/TP is tick-driven (other exit
        # rules for shadows still fire from SMC scanner events as before).
        for sym in shadow_symbols:
            try:
                self._monitor_shadow_position(sym)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[TM] Shadow monitor error {sym}: {e}")
                self._errors += 1

        # Per-trade time-series log (optional).
        try:
            self._sample_trade_logs(time.time())
        except Exception as e:
            if self._errors <= 10:
                print(f"[TM] trade-log sample error: {e}")

    def _sample_trade_logs(self, now: float):
        """Append a {t, price, pnl, mm, mm_dir, exh} sample to each open
        position's `history` every trade_log_interval_sec, when the feature is
        on. Covers real AND shadow positions. The series rides along in the
        position dict → persisted → carried into the closed record on close."""
        s = self._settings
        if not s.get('trade_log_enabled', False):
            return
        try:
            interval = int(s.get('trade_log_interval_sec', 60) or 60)
        except (TypeError, ValueError):
            interval = 60
        interval = max(10, interval)
        # Cheap live maps (single call each), tolerant of FF being absent.
        mm_map, exh_map = {}, {}
        # NOTE(calibration — DO WHEN DATA READY): the ₿ session (btc_dir/
        # btc_paused/btc_mm) is sampled into every chronology point below. When the
        # data is ready, ANALYSE how the ₿ bar behaved DURING each trade's life —
        # did a ₿ flip / WAIT-pause against the position precede drawdowns/losses?
        # did staying aligned with ₿ correlate with winners? This decides (a)
        # whether to gate Queue-2 opens by ₿ (queue2_use_btc) and (b) the deferred
        # question of DROPPING opposite-side signals at signal time when a
        # directional gate (₿ / buttons) is active, instead of queuing them.
        btc_dir, btc_paused, btc_mm = None, None, None
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            if ff:
                mm_map = ff.get_fuel_strength_map() or {}
                exh_map = ff.get_exhaustion_map() or {}
                # ₿ BTCUSDT banner (session) + BTC ММ strength — same for every
                # sample this tick, recorded so the trade chart can overlay them.
                sess = ff.get_btc_session() or {}
                btc_dir = sess.get('dir')
                btc_paused = bool(sess.get('paused'))
                btc_mm = (mm_map.get('BTCUSDT') or {}).get('now')
        except Exception:
            pass
        with self._lock:
            items = list(self._positions.items()) + list(self._shadow_positions.items())
        changed = False
        for sym, pos in items:
            try:
                if (now - float(pos.get('_last_log_at', 0))) < interval:
                    continue
                price = self._get_current_price(sym)
                if not price:
                    continue
                entry = pos.get('entry_price')
                side = pos.get('side')
                if entry and side:
                    pnl = ((price - entry) / entry * 100.0) if side == 'LONG' \
                        else ((entry - price) / entry * 100.0)
                else:
                    pnl = 0.0
                mm = mm_map.get(sym) or {}
                # ⚡ CTR chronology: STC value, last crossover direction, and the
                # crossover age in bars — recorded per sample so the trade chart
                # can plot how the CTR reading (and its crossover age) evolved.
                ctr = self._ctr_snapshot(sym) or {}
                # Live HOLD-score + Manual SL + distance-to-SL for richer replay.
                hold_score = None
                try:
                    _sc = self._ff_score_dict(sym) if hasattr(self, '_ff_score_dict') else None
                    if _sc is None:
                        from detection.fuel_filter import get_fuel_filter
                        _ff2 = get_fuel_filter()
                        _sc = _ff2.score_dict(sym) if _ff2 else None
                    hold_score = (_sc or {}).get('score')
                except Exception:
                    hold_score = None
                # 🤪 Decision Center TRAJECTORY — sampled at a COARSER cadence
                # than price (heavy: LONG+SHORT eval + context) so we can see how
                # the verdict evolved intra-trade without loading the -w1 worker.
                dec_reco = dec_verdict = dec_prob = dec_score = None
                mtf_align = mtf_trend = mtf_timing = None
                try:
                    _dec_iv = int(self._settings.get('trade_log_decision_interval_sec', 180) or 180)
                    if (now - float(pos.get('_last_dec_at', 0))) >= max(60, _dec_iv):
                        _due = True
                        if self._settings.get('entry_score_enabled', True):
                            _dec = self.compute_decision(sym, price)
                            if _dec:
                                dec_reco = _dec.get('recommended')
                                dec_verdict = _dec.get('verdict')
                                dec_prob = (_dec.get('prob_long') if side == 'LONG'
                                            else _dec.get('prob_short'))
                                dec_score = (_dec.get('long_score') if side == 'LONG'
                                             else _dec.get('short_score'))
                        # Multi-TF CTR confluence trajectory (same cadence).
                        try:
                            from detection.fuel_filter import get_fuel_filter
                            _ff3 = get_fuel_filter()
                            _mtf = _ff3._ctr_confluence(sym, side) if _ff3 else None
                            if _mtf:
                                mtf_align = _mtf.get('align')
                                mtf_trend = _mtf.get('trend')
                                mtf_timing = _mtf.get('timing')
                        except Exception:
                            pass
                        pos['_last_dec_at'] = now
                except Exception:
                    pass
                try:
                    _msl = float(pos.get('manual_sl') or 0) or None
                except (TypeError, ValueError):
                    _msl = None
                dist_to_sl = None
                if _msl and price:
                    try:
                        dist_to_sl = round((price - _msl) / price * 100.0
                                           if side == 'LONG'
                                           else (_msl - price) / price * 100.0, 3)
                    except Exception:
                        dist_to_sl = None
                sample = {
                    't': int(now),
                    'price': price,
                    'pnl': round(pnl, 4),
                    'mm': mm.get('now'),
                    'mm_dir': mm.get('dir'),
                    'exh': exh_map.get(sym),
                    # ⚡ CTR at this moment.
                    'ctr_stc': ctr.get('stc'),
                    'ctr_dir': ctr.get('last_dir'),
                    'ctr_age': ctr.get('last_signal_age_bars'),
                    # ⭐ hold-score + Manual SL/TP levels + % distance price→SL.
                    'hold_score': hold_score,
                    'manual_sl': _msl,
                    'manual_tp': pos.get('manual_tp'),
                    'dist_to_sl': dist_to_sl,
                    # 🤪 Decision Center trajectory (coarse cadence; None between).
                    'dec_reco': dec_reco,
                    'dec_verdict': dec_verdict,
                    'dec_prob': dec_prob,
                    'dec_score': dec_score,
                    # ⚡ Multi-TF CTR confluence trajectory (same cadence).
                    'mtf_align': mtf_align,
                    'mtf_trend': mtf_trend,
                    'mtf_timing': mtf_timing,
                    # 🔄 reversal-after-peak readiness % (stamped by the monitor).
                    'rev': pos.get('ctr_rev_pct'),
                    # ₿ BTCUSDT banner state + BTC ММ at this moment.
                    'btc_dir': btc_dir,
                    'btc_paused': btc_paused,
                    'btc_mm': btc_mm,
                }
                with self._lock:
                    hist = pos.setdefault('history', [])
                    # Baseline point at the EXACT open moment (PnL=0, price=entry)
                    # so EVERY trade's chart starts cleanly at 0 — otherwise the
                    # first plotted sample is taken a few seconds AFTER open, by
                    # when price already drifted, so the line starts off-zero.
                    if not hist and pos.get('opened_at') and entry:
                        hist.append({'t': int(pos['opened_at']), 'price': float(entry),
                                     'pnl': 0.0})
                    hist.append(sample)
                    if len(hist) > TRADE_LOG_MAX:
                        del hist[:len(hist) - TRADE_LOG_MAX]
                    pos['_last_log_at'] = now
                changed = True
            except Exception:
                continue
        if changed:
            self._persist_positions()
            self._persist_shadow_positions()

    def get_trade_history(self, symbol: str, closed_at=None,
                          is_shadow: bool = False) -> Dict:
        """Return the recorded time-series for a trade. Matches a CLOSED trade
        by symbol (+ closed_at if given); falls back to the live OPEN position
        so an in-progress trade can be charted too."""
        symbol = (symbol or '').upper()
        src = self._shadow_closed if is_shadow else self._closed_trades
        with self._lock:
            rows = list(src)
            op = (self._shadow_positions if is_shadow
                  else self._positions).get(symbol)
        best = None
        for c in rows:
            if str(c.get('symbol', '')).upper() != symbol:
                continue
            if closed_at is not None:
                try:
                    if abs(float(c.get('closed_at', 0)) - float(closed_at)) > 2:
                        continue
                except (TypeError, ValueError):
                    continue
            best = c   # keep the last (most recent) match
        if best is not None:
            _hist = list(best.get('history') or [])
            # Peak must never contradict the plotted samples. Derive it from the
            # SAME series the chart draws, so already-closed records with a stale
            # stored peak (0.00) display the true MFE. Max of stored / samples /
            # final PnL.
            _pcands = []
            _sp = best.get('peak_pnl_pct')
            if _sp is not None:
                _pcands.append(_sp)
            _fp = best.get('pnl_pct')
            if _fp is not None:
                _pcands.append(_fp)
            for _h in _hist:
                _hp = _h.get('pnl')
                if _hp is not None:
                    _pcands.append(_hp)
            _eff_peak = round(max(_pcands), 4) if _pcands else _sp
            return {'ok': True, 'symbol': symbol, 'open': False,
                    'side': best.get('side'), 'entry_price': best.get('entry_price'),
                    'exit_price': best.get('exit_price'),
                    'pnl_pct': best.get('pnl_pct'),
                    'peak_pnl_pct': _eff_peak,
                    'entry_decision': best.get('entry_score'),
                    'exit_decision': best.get('exit_decision'),
                    # Why the trade was closed — lets the modal show whether the
                    # 🤖 soft trailing-TP (bot_tp_trail) or something else exited.
                    'exit_reason': best.get('reason'),
                    'exit_reason_detail': best.get('reason_detail'),
                    # ⚡ CTR reading at open + at close.
                    'ctr_open': best.get('ctr_open'),
                    'ctr_close': best.get('ctr_close'),
                    'opened_at': best.get('opened_at'),
                    'closed_at': best.get('closed_at'),
                    'history': list(best.get('history') or [])}
        if op:
            return {'ok': True, 'symbol': symbol, 'open': True,
                    'side': op.get('side'), 'entry_price': op.get('entry_price'),
                    'peak_pnl_pct': self._peak_pnl(
                        symbol, op.get('pnl_pct') or 0.0, is_shadow=is_shadow),
                    'entry_decision': op.get('entry_score'),
                    # live Decision Center as of THIS view (heavy — one call)
                    'decision_now': self._decision_snapshot(symbol),
                    'ctr_open': op.get('ctr_open'),
                    'ctr_close': self._ctr_snapshot(symbol),   # "as of now" for open
                    'opened_at': op.get('opened_at'),
                    'now_ts': time.time(),
                    'history': list(op.get('history') or [])}
        return {'ok': True, 'symbol': symbol, 'history': []}

    def _bot_soft_exit(self, pnl_pct: float, peak_pct) -> Optional[str]:
        """🤖 Bot-side SOFT exit layer — the bot watches live PnL% itself and
        decides to close, INDEPENDENTLY of the exchange-native SL/TP/Trailing
        (which stay as a hard backstop). Identical for REAL and PAPER trades.

        Rules (all off by default):
          • soft SL  — fixed: close when pnl% <= -bot_sl_pct.
          • soft TP  — trailing ladder: once the peak reaches bot_tp_pct, start
            trailing; close when pnl retraces bot_tp_giveback_pct below the peak
            (peak 1.5%→stop 1.0%; peak 2.5%→stop 2.0%; …).
        Returns a close reason ('bot_sl' | 'bot_tp_trail') or None."""
        s = self._settings
        if not s.get('bot_exit_enabled'):
            return None
        # Soft Stop Loss (fixed).
        if s.get('bot_sl_enabled'):
            try:
                slp = float(s.get('bot_sl_pct', 0) or 0)
            except (TypeError, ValueError):
                slp = 0.0
            if slp > 0 and pnl_pct <= -slp:
                return 'bot_sl'
        # Soft trailing Take Profit (ladder).
        if s.get('bot_tp_enabled') and peak_pct is not None:
            try:
                tpp = float(s.get('bot_tp_pct', 0) or 0)
                gb = float(s.get('bot_tp_giveback_pct', 0.5) or 0.5)
            except (TypeError, ValueError):
                tpp, gb = 0.0, 0.5
            # Trailing is active only once the peak has reached the activation.
            if tpp > 0 and peak_pct >= tpp and pnl_pct <= (peak_pct - gb):
                return 'bot_tp_trail'
        return None

    @staticmethod
    def _tf_to_seconds(tf) -> int:
        """Parse a timeframe string ('15m','1h','4h','1d') to seconds. 15m default."""
        try:
            t = str(tf).lower().strip()
            if t.endswith('m'):
                return max(60, int(t[:-1]) * 60)
            if t.endswith('h'):
                return int(t[:-1]) * 3600
            if t.endswith('d'):
                return int(t[:-1]) * 86400
        except Exception:
            pass
        return 900

    def _ctr_reversal_eval(self, symbol: str, side: str, pnl_pct, peak_pct):
        """🔄 CTR reversal-after-peak evaluator. Returns (rev_pct, hard_ok, tf_sec):
          • rev_pct  0..100 — live readiness to close (for the open-positions
            column): blends CTR opposite-lean, PnL give-back, exhaustion, ММ;
          • hard_ok — the HARD rule is TRUE right now: a real peak happened AND
            CTR is in the opposite extreme AND ≥N of 3 confirmations of a decline.
            (The caller adds a persistence-in-bars guard before closing.)
          • tf_sec — the CTR timeframe in seconds (for the bars persistence guard).
        Uses only cheap cached reads (forecast CTR + FF maps)."""
        s = self._settings
        try:
            min_peak = float(s.get('ctr_reversal_min_peak_pct', 1.0) or 0)
            ctr_need = float(s.get('ctr_reversal_ctr_pct', 70) or 0)
            gb_need = float(s.get('ctr_reversal_giveback_pct', 30) or 0)
            exh_need = float(s.get('ctr_reversal_exh_pct', 66) or 0)
            conf_need = int(s.get('ctr_reversal_confirmations', 2) or 2)
            min_lock = float(s.get('ctr_reversal_min_lock_pct', 0.0))
        except (TypeError, ValueError):
            min_peak, ctr_need, gb_need, exh_need, conf_need, min_lock = 1.0, 70.0, 30.0, 66.0, 2, 0.0
        opp = 'SHORT' if side == 'LONG' else 'LONG'
        # 1) CTR opposite-extreme strength (0..100).
        ctr_against = 0.0
        ctr = self._ctr_snapshot(symbol) or {}
        stc = ctr.get('stc')
        if stc is not None:
            try:
                stc = float(stc)
                lean = 'SHORT' if stc > 50 else ('LONG' if stc < 50 else None)
                if lean == opp:
                    ctr_against = abs(stc - 50.0) / 50.0 * 100.0
            except (TypeError, ValueError):
                pass
        # 2) Confirmations.
        conf = 0
        # (a) give-back from peak.
        gb_ratio = 0.0
        if peak_pct and peak_pct > 0 and pnl_pct is not None and gb_need > 0:
            given = peak_pct - pnl_pct
            gb_ratio = max(0.0, min(1.0, given / (peak_pct * gb_need / 100.0)))
            if given >= peak_pct * gb_need / 100.0:
                conf += 1
        # (b) exhaustion + (c) ММ against/weak — from the cheap FF maps.
        exh, mm = None, None
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            if ff:
                exh = ff.get_exhaustion_map().get(symbol)
                mm = ff.get_fuel_strength_map().get(symbol)
        except Exception:
            pass
        if exh is not None and exh >= exh_need:
            conf += 1
        mm_against = 0.0
        if mm:
            if mm.get('dir') == opp:
                mm_against = 100.0
            elif mm.get('now') is not None and mm.get('now') < 20:
                mm_against = 60.0
            if mm_against >= 60:
                conf += 1
        # Readiness % (weighted) for the table.
        rev_pct = round(0.55 * ctr_against
                        + 0.20 * gb_ratio * 100.0
                        + 0.15 * (float(exh) if exh is not None else 0.0)
                        + 0.10 * mm_against)
        rev_pct = max(0, min(100, rev_pct))
        # 🔒 Profit-LOCK only: the reversal exit must NOT realise a loss (that is
        # the SL's job). If price already gave the whole move back into the red,
        # there's nothing to «lock» — closing then just sells the bottom (the
        # SLXUSDT case: closed −2.55% right before it recovered +3.6%). So require
        # the CURRENT PnL to still be ≥ min_lock (default 0 = never at a loss).
        _in_lock = (pnl_pct is not None and pnl_pct >= min_lock)
        hard_ok = bool(s.get('use_ctr_reversal_exit', True)
                       and peak_pct is not None and peak_pct >= min_peak
                       and ctr_against >= ctr_need
                       and conf >= conf_need
                       and _in_lock)
        tf_sec = self._tf_to_seconds(ctr.get('tf'))
        return rev_pct, hard_ok, tf_sec

    def _rev_persisted(self, pos: Dict, hard_ok: bool, tf_sec) -> bool:
        """Persistence-in-bars guard for the 🔄 reversal exit. Returns True only
        when `hard_ok` has held CONTINUOUSLY for ≥ ctr_reversal_min_bars bars of
        the CTR timeframe. Any drop resets the timer, so a brief intrabar wobble
        never accumulates toward a close (protects the trade's potential)."""
        if not hard_ok:
            pos.pop('_rev_ready_since', None)
            return False
        try:
            min_bars = int(self._settings.get('ctr_reversal_min_bars', 2) or 0)
        except (TypeError, ValueError):
            min_bars = 2
        now = time.time()
        since = pos.get('_rev_ready_since')
        if not since:
            pos['_rev_ready_since'] = now
            since = now
        if min_bars <= 0:
            return True
        return (now - since) >= min_bars * max(60, int(tf_sec or 900))

    def _monitor_position(self, symbol: str):
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return
        
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return
        
        # Update Health Score evaluator state — peak PnL high-water mark.
        # This is independent of any trailing/BE logic and never decreases.
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            current_pnl_pct = (current_price - entry) / entry * 100
        else:
            current_pnl_pct = (entry - current_price) / entry * 100
        with self._lock:
            st = self._pos_state.get(symbol)
            if st is not None and current_pnl_pct > st.get('peak_pnl_pct', 0):
                st['peak_pnl_pct'] = current_pnl_pct
            # Mirror the peak onto the persisted position dict so it survives a
            # restart (the ephemeral _pos_state is rebuilt on load).
            if current_pnl_pct > pos.get('peak_pnl_pct', 0):
                pos['peak_pnl_pct'] = round(current_pnl_pct, 4)
            # MAE — max adverse excursion (worst drawdown), for SL/risk analytics.
            if current_pnl_pct < pos.get('mae_pnl_pct', 0):
                pos['mae_pnl_pct'] = round(current_pnl_pct, 4)

        # Backfill entry snapshots that were unavailable at the exact open moment
        # (CTR / Decision data can lag the open by a tick). Filled ONCE — so the
        # trade-history modal shows «на вході» for CTR and Decision Center on
        # EVERY trade, not only those opened when the data was already warm.
        if pos.get('ctr_open') is None:
            _c = self._ctr_snapshot(symbol)
            if _c:
                pos['ctr_open'] = _c
        if pos.get('entry_score') is None:
            _d = self.compute_decision(symbol, current_price)
            if _d:
                pos['entry_score'] = _d

        # === Manual SL/TP (per-position overrides set via UI) ===
        # These are user-entered absolute price levels (separate from the
        # position's automatic sl_price/tp_price). They fire FIRST so a
        # manually-set tight stop takes precedence over the strategy's
        # original SL. Empty / 0 / None → not active for that field.
        # Auto-manage the Manual SL from the «Require OB Match» OB first (no-op
        # unless q2_auto_ob_sl is on) so the breach check below sees the latest.
        self._auto_ob_manual_sl(symbol, pos, current_price)
        manual_reason = self._check_manual_sl_tp(pos, current_price)
        if manual_reason:
            self._close_position(symbol, current_price, reason=manual_reason)
            return
        
        # === Manual mode gate ===
        # When the user flipped this position into manual mode, ONLY the
        # manual SL/TP above and force-close from UI can close it. All
        # automatic exit logic and management actions below are bypassed.
        # The position lives or dies by the operator's hand.
        if pos.get('manual_mode'):
            return

        # === 🤖 Bot-side SOFT exit (independent of exchange SL/TP) ===
        with self._lock:
            _st = self._pos_state.get(symbol)
            _peak = _st.get('peak_pnl_pct') if _st else None
        soft_reason = self._bot_soft_exit(current_pnl_pct, _peak)
        if soft_reason:
            self._close_position(symbol, current_price, reason=soft_reason)
            return

        # === 🔄 CTR reversal-after-peak: stamp readiness % + close if fired ===
        # Persistence-in-bars guard: the reversal must hold for ≥ N CTR bars
        # before closing, so a tiny wobble can't exit and give back the move.
        _rev_pct, _rev_hard, _tf_sec = self._ctr_reversal_eval(symbol, pos['side'], current_pnl_pct, _peak)
        pos['ctr_rev_pct'] = _rev_pct
        if self._rev_persisted(pos, _rev_hard, _tf_sec):
            self._close_position(symbol, current_price, reason='ctr_reversal_after_peak')
            return

        # 1) Hard exits — Stop Loss / Take Profit
        s = self._settings
        if s.get('use_sl') and pos.get('sl_price'):
            if (pos['side'] == 'LONG' and current_price <= pos['sl_price']) or \
               (pos['side'] == 'SHORT' and current_price >= pos['sl_price']):
                self._close_position(symbol, current_price, reason='stop_loss')
                return
        
        if s.get('use_tp') and pos.get('tp_price'):
            if (pos['side'] == 'LONG' and current_price >= pos['tp_price']) or \
               (pos['side'] == 'SHORT' and current_price <= pos['tp_price']):
                self._close_position(symbol, current_price, reason='take_profit')
                return
        
        # 2) Time stop
        if s.get('use_time_stop'):
            elapsed_h = (time.time() - pos['opened_at']) / 3600
            if elapsed_h >= s.get('time_stop_hours', 4):
                self._close_position(symbol, current_price, reason='time_stop')
                return
        
        # 3) HTF flip
        if s.get('use_htf_flip') and self.scanner:
            try:
                htf_settings = self.scanner.get_htf_settings()
                if htf_settings.get('enabled'):
                    htf_bias = self.scanner._htf_cache.get(symbol, {}).get('bias', 'neutral')
                    pos_dir = 'bull' if pos['side'] == 'LONG' else 'bear'
                    opposite = 'bear' if pos_dir == 'bull' else 'bull'
                    if htf_bias == opposite:
                        self._close_position(symbol, current_price, reason='htf_flip')
                        return
            except:
                pass
        
        # === Position-management actions (no closes) ===
        self._update_trailing(pos, current_price)
        self._update_be(pos, current_price)
    
    def _check_manual_sl_tp(self, pos: Dict,
                              current_price: float) -> Optional[str]:
        """Return 'manual_sl' / 'manual_tp' / None depending on whether the
        user-entered manual stop level has been breached. Works for both
        real and shadow positions — the only thing that matters is the
        position dict shape.
        
        Manual levels are absolute prices (not %). Empty / 0 / None / non-
        numeric values are treated as 'not set' and never trigger. This
        lets the UI represent 'no manual stop' by simply leaving the field
        blank, which is the most natural input pattern.
        
        Manual SL fires when price crosses the configured level AGAINST
        the position:
            LONG  · manual_sl: price ≤ level
            LONG  · manual_tp: price ≥ level
            SHORT · manual_sl: price ≥ level
            SHORT · manual_tp: price ≤ level
        """
        def _to_float(v):
            try:
                f = float(v)
                return f if f > 0 else None
            except (TypeError, ValueError):
                return None
        
        manual_sl = _to_float(pos.get('manual_sl'))
        manual_tp = _to_float(pos.get('manual_tp'))
        side = pos.get('side')
        
        if manual_sl is not None:
            hit = ((side == 'LONG' and current_price <= manual_sl)
                   or (side == 'SHORT' and current_price >= manual_sl))
            if hit:
                return 'manual_sl'
        
        if manual_tp is not None:
            hit = ((side == 'LONG' and current_price >= manual_tp)
                   or (side == 'SHORT' and current_price <= manual_tp))
            if hit:
                return 'manual_tp'

        return None

    def _auto_ob_manual_sl(self, symbol: str, pos: Dict, current_price: float) -> None:
        """Auto-manage this position's Manual SL from the nearest «Require OB
        Match» Order Block (FF setting `q2_auto_ob_sl`). Runs every tick BEFORE
        `_check_manual_sl_tp`, so once the level is written the existing manual-SL
        breach logic closes the trade.

            LONG  → SL = OB low  × (1 − buffer%)   sits BELOW price · ratchets UP
            SHORT → SL = OB high × (1 + buffer%)   sits ABOVE price · ratchets DOWN

        Rules:
          • SET ONCE — the auto SL is written a SINGLE time (the first valid OB
            level on the safe side of price). It is NEVER moved again afterwards
            (NO ratchet on newer OBs). If the FIRST attempt can't set a level
            yet, we KEEP RETRYING each tick until one valid level is placed.
          • The OB is the same «Require OB Match» row that gates entries
            (`get_smc_ob_state(symbol, ob_filter_timeframe)`), matching direction
            (LONG needs a BULLISH OB, SHORT a BEARISH one).
          • Respects a manual SL the operator typed by hand: we only overwrite a
            level we ourselves last wrote (tracked in `pos['_auto_ob_sl_val']`).
            If `manual_sl` differs from that, the user set it → we back off.

        Order of checks (per operator's rule):
          1. An auto SL already placed (`_auto_ob_sl_val` set) → do NOTHING
             (set once, no re-fire).
          2. No auto SL yet → try to determine one from the OB and set it.
          3. A record is written to the 🧾 log ONLY when, after all checks, the
             Manual SL field is STILL EMPTY (not a single level could be set) —
             then we log the reason. Any SL in place ⇒ silence."""
        try:
            from detection.fuel_filter import get_fuel_filter
            s = get_fuel_filter().get_settings()
        except Exception:
            return
        if not s.get('q2_auto_ob_sl'):
            return
        try:
            buf = float(s.get('q2_auto_ob_sl_buffer_pct', 0.2) or 0.0) / 100.0
        except (TypeError, ValueError):
            buf = 0.002
        buf = max(0.0, buf)

        side = pos.get('side')
        try:
            cur_sl = float(pos.get('manual_sl') or 0) or None
        except (TypeError, ValueError):
            cur_sl = None
        auto_val = pos.get('_auto_ob_sl_val')

        # A Manual SL the operator typed by hand → never touch, never log
        # (the field is NOT empty, so there is nothing to report).
        if cur_sl is not None and (auto_val is None or abs(cur_sl - float(auto_val)) > 1e-12):
            return

        # 🔒 SET ONCE: as soon as we've successfully placed an auto SL from an OB,
        # we never move it again (no ratchet on newer OBs). Retries continue ONLY
        # until the FIRST successful set — so a trade that couldn't get a level
        # yet still gets one when a valid OB finally appears.
        if auto_val is not None:
            return

        def _diag(reason: str):
            """Write a skip-reason to the 🧾 log ONLY when the Manual SL field is
            left EMPTY after all checks (no user SL, no auto SL could be set). If
            ANY level is already in place we make NO record — per the rule «запис
            лише коли лишили пусте поле Manual SL». Throttled to once per 5 min."""
            try:
                if pos.get('manual_sl'):      # field NOT empty → make no record
                    return
                nowt = time.time()
                # Throttle to once per 30 min/position — «OB не готовий» повторюється
                # щоскан і забивав 🧾 лог (особливо коли SMC-сканер ще рахує OB).
                if nowt - float(pos.get('_auto_ob_sl_diag_at') or 0) < 1800:
                    return
                pos['_auto_ob_sl_diag_at'] = nowt
                from detection.activity_log import log_activity
                log_activity(symbol, 'skipped', f"Авто-SL з OB: {reason}",
                             side=side, source='TM')
            except Exception:
                pass

        # OB timeframe is Queue-2-specific (its own setting, default 15m — the
        # main scan TF, always computed). NOT the scanner's ob_filter_timeframe.
        ob_tf = str(s.get('q2_auto_ob_sl_tf', '15m') or '15m').lower()
        try:
            from storage.db_operations import get_db
            row = get_db().get_smc_ob_state(symbol, ob_tf)
        except Exception:
            return
        if not row or not row.get('bias'):
            _diag(f"немає готового OB на {ob_tf.upper()} (сканер ще не порахував)")
            return
        bias = row.get('bias')
        try:
            bar_low = float(row.get('bar_low'))
            bar_high = float(row.get('bar_high'))
        except (TypeError, ValueError):
            return

        if side == 'LONG':
            if bias != 'BULLISH':
                _diag(f"OB на {ob_tf.upper()} протилежний (BEARISH) — чекаю BULLISH")
                return
            cand = bar_low * (1.0 - buf)
            if cand >= current_price:
                _diag(f"низ OB {self._fmt_price(bar_low)} вище ціни "
                      f"{self._fmt_price(current_price)} — SL там закрив би угоду")
                return   # OB not below price → wouldn't protect, skip
        elif side == 'SHORT':
            if bias != 'BEARISH':
                _diag(f"OB на {ob_tf.upper()} протилежний (BULLISH) — чекаю BEARISH")
                return
            cand = bar_high * (1.0 + buf)
            if cand <= current_price:
                _diag(f"верх OB {self._fmt_price(bar_high)} нижче ціни "
                      f"{self._fmt_price(current_price)} — SL там закрив би угоду")
                return   # OB not above price → wouldn't protect, skip
        else:
            return

        # First (and only) successful placement — set once, never ratcheted after.
        cand = self._round_sltp_value(cand)   # clean value (no float tail)
        pos['manual_sl'] = cand
        pos['_auto_ob_sl_val'] = cand
        self._record_manual_hist(pos, 'sl', cand)
        try:
            from detection.activity_log import log_activity
            log_activity(symbol, 'autosl',
                         f"SL встановлено з OB {ob_tf.upper()} → {self._fmt_price(cand)} "
                         f"(буфер {s.get('q2_auto_ob_sl_buffer_pct', 0.2)}%, один раз)",
                         side=side, source='TM')
        except Exception:
            pass

    def _monitor_shadow_position(self, symbol: str):
        """Per-tick monitor for shadow positions — currently only enforces
        manual SL/TP overrides. The strategy's automatic SL/TP and other
        exit rules (HTF flip, reverse SMC, etc.) for shadow positions are
        still driven by SMC scanner events, not by this tick — that's the
        existing design and we leave it alone. Manual levels are the only
        thing that NEEDS a tick because they have no scanner-event hook."""
        with self._lock:
            pos = self._shadow_positions.get(symbol)
        if not pos:
            return
        
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return
        
        # Peak PnL high-water mark for the shadow position (mirrors the real
        # path). Without this, paper trades never accumulated a peak and the
        # close detail could show a peak below the realised PnL.
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            cur_pnl = (current_price - entry) / entry * 100
        else:
            cur_pnl = (entry - current_price) / entry * 100
        with self._lock:
            sst = self._shadow_pos_state.get(symbol)
            if sst is not None and cur_pnl > sst.get('peak_pnl_pct', 0):
                sst['peak_pnl_pct'] = cur_pnl
            # Mirror onto the persisted shadow-position dict (survives restart).
            if cur_pnl > pos.get('peak_pnl_pct', 0):
                pos['peak_pnl_pct'] = round(cur_pnl, 4)
            # MAE — max adverse excursion (worst drawdown), for SL/risk analytics.
            if cur_pnl < pos.get('mae_pnl_pct', 0):
                pos['mae_pnl_pct'] = round(cur_pnl, 4)
            _peak = sst.get('peak_pnl_pct') if sst else None

        # Backfill entry snapshots that were unavailable at the exact open moment
        # (CTR / Decision data can lag the open by a tick) — filled ONCE so the
        # trade-history modal shows «на вході» for CTR + Decision on paper trades.
        if pos.get('ctr_open') is None:
            _c = self._ctr_snapshot(symbol)
            if _c:
                pos['ctr_open'] = _c
        if pos.get('entry_score') is None:
            _d = self.compute_decision(symbol, current_price)
            if _d:
                pos['entry_score'] = _d

        self._auto_ob_manual_sl(symbol, pos, current_price)
        manual_reason = self._check_manual_sl_tp(pos, current_price)
        if manual_reason:
            self._close_shadow(symbol, current_price, reason=manual_reason)
            return

        # === 🤖 Bot-side SOFT exit (same rules as real; paper computes PnL) ===
        if not pos.get('manual_mode'):
            soft_reason = self._bot_soft_exit(cur_pnl, _peak)
            if soft_reason:
                self._close_shadow(symbol, current_price, reason=soft_reason)
                return

        # === 🔄 CTR reversal-after-peak: stamp readiness % + close if fired ===
        _rev_pct, _rev_hard, _tf_sec = self._ctr_reversal_eval(symbol, pos['side'], cur_pnl, _peak)
        pos['ctr_rev_pct'] = _rev_pct
        if not pos.get('manual_mode') and self._rev_persisted(pos, _rev_hard, _tf_sec):
            self._close_shadow(symbol, current_price, reason='ctr_reversal_after_peak')
            return

        # === Automatic exits — paper now MIRRORS the real book so paper PnL ===
        # === reflects what a live trade would actually do (SL/TP/trailing/BE/  ===
        # === time-stop/HTF-flip). Bypassed in manual mode, same as real.      ===
        if pos.get('manual_mode'):
            return
        s = self._settings
        # 1) Hard Stop Loss / Take Profit.
        if s.get('use_sl') and pos.get('sl_price'):
            if (pos['side'] == 'LONG' and current_price <= pos['sl_price']) or \
               (pos['side'] == 'SHORT' and current_price >= pos['sl_price']):
                self._close_shadow(symbol, current_price, reason='stop_loss')
                return
        if s.get('use_tp') and pos.get('tp_price'):
            if (pos['side'] == 'LONG' and current_price >= pos['tp_price']) or \
               (pos['side'] == 'SHORT' and current_price <= pos['tp_price']):
                self._close_shadow(symbol, current_price, reason='take_profit')
                return
        # 2) Time stop.
        if s.get('use_time_stop'):
            elapsed_h = (time.time() - pos['opened_at']) / 3600
            if elapsed_h >= s.get('time_stop_hours', 4):
                self._close_shadow(symbol, current_price, reason='time_stop')
                return
        # 3) HTF flip.
        if s.get('use_htf_flip') and self.scanner:
            try:
                htf_settings = self.scanner.get_htf_settings()
                if htf_settings.get('enabled'):
                    htf_bias = self.scanner._htf_cache.get(symbol, {}).get('bias', 'neutral')
                    pos_dir = 'bull' if pos['side'] == 'LONG' else 'bear'
                    opposite = 'bear' if pos_dir == 'bull' else 'bull'
                    if htf_bias == opposite:
                        self._close_shadow(symbol, current_price, reason='htf_flip')
                        return
            except Exception:
                pass
        # 4) Trailing stop + Break-even (paper-aware — no exchange calls).
        self._update_trailing(pos, current_price, is_shadow=True)
        self._update_be(pos, current_price, is_shadow=True)

    def _update_trailing(self, pos, current_price, is_shadow=False):
        s = self._settings
        _close = self._close_shadow if is_shadow else self._close_position
        # Two paths can activate trailing:
        #   (a) Global use_trailing=True with price reaching trailing_activate_pct
        #   (b) BOS-2 hook explicitly setting trailing_via_bos2=True
        # Path (b) must work even when use_trailing is OFF — that's the
        # explicit promise of the "Auto-activate Trailing after BOS-2"
        # toggle. Otherwise users who disable global trailing but enable
        # BOS-2 trailing would see no effect, which broke an entire
        # trade-management feature.
        global_trailing = s.get('use_trailing', False)
        bos2_trailing = pos.get('trailing_via_bos2', False) and pos.get('trailing_active', False)
        if not global_trailing and not bos2_trailing:
            return
        
        entry = pos['entry_price']
        side = pos['side']
        
        # Activation — only via the global path. The BOS-2 path activates
        # in on_bos_event by setting trailing_active and trailing_via_bos2.
        if not pos.get('trailing_active') and global_trailing:
            activate_pct = s.get('trailing_activate_pct', 1.0) / 100
            if side == 'LONG' and current_price >= entry * (1 + activate_pct):
                pos['trailing_active'] = True
                pos['trailing_peak'] = current_price
            elif side == 'SHORT' and current_price <= entry * (1 - activate_pct):
                pos['trailing_active'] = True
                pos['trailing_peak'] = current_price
        
        if not pos.get('trailing_active'):
            return
        
        # Track peak
        if side == 'LONG' and current_price > pos['trailing_peak']:
            pos['trailing_peak'] = current_price
        elif side == 'SHORT' and current_price < pos['trailing_peak']:
            pos['trailing_peak'] = current_price
        
        # Check if price retraced past trailing distance
        dist_pct = s.get('trailing_distance_pct', 0.5) / 100
        if side == 'LONG':
            trail_stop = pos['trailing_peak'] * (1 - dist_pct)
            if current_price <= trail_stop:
                _close(pos['symbol'], current_price, reason='trailing_stop')
        else:
            trail_stop = pos['trailing_peak'] * (1 + dist_pct)
            if current_price >= trail_stop:
                _close(pos['symbol'], current_price, reason='trailing_stop')
    
    def _update_be(self, pos, current_price, is_shadow=False):
        s = self._settings
        if not s.get('use_be') or pos.get('be_moved'):
            return

        trigger = s.get('be_trigger_pct', 0.5) / 100
        # Commission buffer — SL is placed slightly past entry so when it
        # triggers, the close covers round-trip trading fees instead of
        # leaving the user with a small loss. Default 0.12% ≈ Bybit Perp
        # round-trip taker (0.06% open + 0.06% close). Setting to 0 gives
        # the legacy behavior (SL exactly at entry).
        buf = s.get('be_commission_buffer_pct', 0.12) / 100
        entry = pos['entry_price']
        
        if pos['side'] == 'LONG' and current_price >= entry * (1 + trigger):
            new_sl = entry * (1 + buf)
            pos['sl_price'] = new_sl
            pos['be_moved'] = True
            if not is_shadow:
                self._update_exchange_sl(pos['symbol'], new_sl)
            buf_note = f" (+{buf*100:.2f}% buffer)" if buf > 0 else ""
            self._notify(f"⚖️ BE{' [paper]' if is_shadow else ''}: SL → {self._fmt_price(new_sl)} for "
                          f"{pos['symbol']}{buf_note}")
            (self._persist_shadow_positions if is_shadow else self._persist_positions)()
        elif pos['side'] == 'SHORT' and current_price <= entry * (1 - trigger):
            new_sl = entry * (1 - buf)
            pos['sl_price'] = new_sl
            pos['be_moved'] = True
            if not is_shadow:
                self._update_exchange_sl(pos['symbol'], new_sl)
            buf_note = f" (−{buf*100:.2f}% buffer)" if buf > 0 else ""
            self._notify(f"⚖️ BE{' [paper]' if is_shadow else ''}: SL → {self._fmt_price(new_sl)} for "
                          f"{pos['symbol']}{buf_note}")
            (self._persist_shadow_positions if is_shadow else self._persist_positions)()

    # ============================================================
    # Signal hooks (called from SMC scanner)
    # ============================================================
    
    def on_signal(self, symbol: str, side: str, entry_price: float, opened_by: str):
        """Called when SMC scanner fires a signal.
        side: 'LONG' or 'SHORT'
        opened_by: 'choch' or 'choch_bos'
        
        Behavior matrix:
          - No existing position: open (real if enabled, shadow if test_mode)
          - Same-direction position exists: ignore (dedup at signal level)
          - Opposite-direction position exists: REVERSE — close existing
            with reason='reverse_signal' first, then open the new one.
            This is distinct from on_choch_event Reverse SMC: that fires
            on EVERY CHoCH (no filters). reverse_signal fires only when
            the new signal has cleared ALL gates (dedup, HTF, OB filter,
            mode-specific recency) — i.e. it's a "qualified" reversal,
            same caliber as a fresh entry.
        
        The reverse path runs in BOTH real and shadow modes so paper-trading
        produces the same trade history as a live deployment would.

        Returns a status dict the SMC scanner uses to place the correct chart
        marker and surface the real reason (never None):
          {'real_opened': bool, 'shadow_opened': bool,
           'status': 'opened'|'rejected'|'duplicate',
           'reason': str, 'is_paper': bool}
        """
        s = self._settings
        enabled = self.is_enabled()
        test_mode = s.get('test_mode', True)
        # Auto-reverse on an opposite QUALIFIED entry signal (close + reopen the
        # other way). This is SEPARATE from the "Reverse SMC (CHoCH only)" exit
        # rule (use_reverse_smc), which only CLOSES on an opposite CHoCH. OFF by
        # default so disabling the Exit Rules actually holds the position instead
        # of flipping it on the next opposite signal.
        allow_reverse = bool(s.get('use_reverse_signal', False))

        try:
            from detection.activity_log import log_activity
        except Exception:
            log_activity = lambda *a, **k: None
        # NB: the 'signal' arrival event is logged EARLIER, in the scanner's
        # _send_alert (before its OB/PD/Forecast filters), so it isn't repeated
        # here. This method logs only the OUTCOME (queued/opened/skipped/…).

        with self._lock:
            existing_real = self._positions.get(symbol)
            existing_shadow = self._shadow_positions.get(symbol)

        # === Manual mode gate ===
        # If the user has locked this symbol into manual mode on EITHER
        # real or shadow track, ignore new signals entirely — no open, no
        # reverse. The operator wants to manage this trade by hand.
        if existing_real and existing_real.get('manual_mode'):
            print(f"[TM] {symbol} in manual mode (real) — signal ignored")
            log_activity(symbol, 'skipped', 'Ручний режим (real) — сигнал проігноровано', side=side, source='TM')
            return {'status': 'duplicate', 'reason': 'manual mode (real) — signal ignored'}
        if existing_shadow and existing_shadow.get('manual_mode'):
            print(f"[TM] {symbol} in manual mode (shadow) — signal ignored")
            log_activity(symbol, 'skipped', 'Ручний режим (paper) — сигнал проігноровано', side=side, source='TM')
            return {'status': 'duplicate', 'reason': 'manual mode (shadow) — signal ignored'}

        # === Fuel Auto-Filter interception ===
        # While FF is ON, every coin the main flow would open is QUEUED in the
        # ❤️ FF base instead of opening now. (FF's own opens use bypass_gates and
        # never reach here.)
        #
        # Routing rule:
        #   • no position                     → queue the signal.
        #   • OPPOSITE position + «реверс через Чергу-2» ON → queue it too, so the
        #     reversal is decided by Queue 2's FULL algorithm (SCORE+CTR). The Q2
        #     engine closes the opposite trade only when the signal is ready to
        #     open — NOT on the bare signal.
        #   • SAME-dir position               → NOT queued → handled as duplicate.
        _pos = existing_real or existing_shadow
        _pos_side = _pos.get('side') if _pos else None
        _route_ff = (_pos is None)
        if _pos is not None and _pos_side and _pos_side != side:
            try:
                from detection.fuel_filter import get_fuel_filter
                _ffx = get_fuel_filter()
                if _ffx and _ffx.is_enabled():
                    _fs = _ffx.get_settings()
                    if _fs.get('queue2_enabled') and _fs.get('queue2_reverse_via_queue', True):
                        _route_ff = True   # opposite → let Queue 2 gate the reverse
            except Exception:
                pass
        if _route_ff:
            try:
                from detection.fuel_filter import get_fuel_filter
                ff = get_fuel_filter()
                if ff and ff.is_enabled():
                    _disp = ff.intercept(symbol, side, kind=opened_by)
                    if _disp == 'queued':
                        _rev = ' (реверс — коли пройде Чергу-2)' if _pos is not None else ''
                        return {'status': 'queued', 'is_paper': False,
                                'reason': f'у Черзі ❤️ Fuel Auto-Filter (чекає фільтр){_rev}'}
                    if _disp == 'dropped':
                        # An enabled queue (Q2) OWNED but REJECTED it (CTR gate) —
                        # NOT queued, NOT opened (and NOT reversed). The marker must
                        # say so, not lie «queued». (Q1 off ≠ signal goes to Q1.)
                        return {'status': 'rejected', 'is_paper': False,
                                'reason': 'Черга-2 відкинула: CTR-нахил не в бік сигналу'}
                    # '' → both queues OFF → falls through to a direct open below.
                else:
                    log_activity(symbol, 'skipped', 'Fuel Auto-Filter вимкнено — сигнал іде повз черги', side=side, source='TM')
            except Exception as e:
                # An intercept EXCEPTION was the silent «signal disappears» bug —
                # now it's always recorded instead of vanishing.
                print(f"[TM] FF intercept error for {symbol}: {e}")
                log_activity(symbol, 'rejected', f'Помилка перехоплення Fuel-фільтром: {e}', side=side, source='TM')

        # === Real-money track ===
        # Runs whenever TM is enabled. Gated by the tradeable list AND max_open_positions.
        # If limit reached: signal goes to paper track instead (if test_mode on).
        real_opened = False
        real_reason = ''
        at_limit = self._at_max_positions()

        if enabled and not at_limit:
            if existing_real:
                if existing_real['side'] == side:
                    # Same direction — already in this trend, no new trade.
                    log_activity(symbol, 'skipped', 'Вже в угоді (real, той самий напрямок) — сигнал ігнорується', side=side, source='TM')
                    return {'real_opened': False, 'status': 'duplicate',
                            'reason': f'already holding {side} (real) — no new trade'}
                else:
                    # OPPOSITE direction — flip only if auto-reverse is enabled.
                    if not allow_reverse:
                        print(f"[TM] {symbol}: opposite signal but reverse-on-signal "
                              f"OFF — holding {existing_real['side']}")
                        log_activity(symbol, 'skipped', f'Протилежний сигнал, але «Reverse on opposite signal» ВИМКНЕНО — тримаємо {existing_real["side"]} (real)', side=side, source='TM')
                        return {'status': 'duplicate', 'reason':
                                f'opposite signal, but "Reverse on opposite signal" '
                                f'is OFF — holding {existing_real["side"]} position'}
                    print(f"[TM] 🔄 Reverse signal for {symbol}: "
                          f"closing {existing_real['side']} → opening {side}")
                    try:
                        self._close_position(symbol, entry_price, reason='reverse_signal')
                    except Exception as e:
                        print(f"[TM] ❌ Reverse-close failed for {symbol}: {e}")
                        return {'real_opened': False, 'status': 'rejected',
                                'reason': f'reverse-close failed: {e}'}
                    if self._is_tradeable(symbol):
                        res = self._open_position(symbol, side, entry_price, opened_by) or {}
                        if res.get('ok'):
                            real_opened = True
                        else:
                            real_reason = res.get('reason', 'reverse-open blocked')
                    else:
                        real_reason = f'{symbol} not in tradeable list (reverse-open skipped)'
            else:
                # No existing real position
                if self._is_tradeable(symbol):
                    res = self._open_position(symbol, side, entry_price, opened_by) or {}
                    if res.get('ok'):
                        real_opened = True
                    else:
                        real_reason = res.get('reason', 'open blocked')
                else:
                    real_reason = f'{symbol} not in tradeable list (real)'
        elif enabled and at_limit:
            # Max positions reached — log it but don't block paper track
            real_reason = (f'max open positions '
                           f'({self._settings.get("max_open_positions", 10)}) reached')
            print(f"[TM] ℹ️ {real_reason}, {symbol} signal → Test Mode")

        if real_opened:
            log_activity(symbol, 'opened', 'Відкрито реальну позицію (TM)', side=side, source='TM')
            return {'real_opened': True, 'status': 'opened', 'is_paper': False}

        # === Paper (shadow) track ===
        # Independent of the real track. Runs on EVERY qualified signal
        # whenever test_mode is on — there is no tradeable gate here because
        # paper trades never touch real funds; the whole point is to measure
        # how the FULL strategy would have performed. The one exception: if
        # the real track just opened/holds this exact symbol, we skip the
        # shadow to avoid a redundant duplicate of a position we're already
        # tracking for real (the real trade is the source of truth for it).
        if test_mode and not real_opened:
            if existing_shadow:
                if existing_shadow['side'] == side:
                    log_activity(symbol, 'skipped', 'Вже в угоді (paper, той самий напрямок) — сигнал ігнорується', side=side, source='TM')
                    return {'status': 'duplicate',
                            'reason': f'already holding {side} (paper) — no new trade'}
                # OPPOSITE direction — flip only if auto-reverse is enabled.
                if not allow_reverse:
                    log_activity(symbol, 'skipped', f'Протилежний сигнал, але «Reverse on opposite signal» ВИМКНЕНО — тримаємо {existing_shadow["side"]} (paper)', side=side, source='TM')
                    return {'status': 'duplicate', 'reason':
                            f'opposite signal, but "Reverse on opposite signal" is '
                            f'OFF — holding {existing_shadow["side"]} paper position'}
                print(f"[TM] 🔄 [TEST] Reverse signal for {symbol}: "
                      f"closing {existing_shadow['side']} → opening {side}")
                try:
                    self._close_shadow(symbol, entry_price, reason='reverse_signal')
                except Exception as e:
                    print(f"[TM] ❌ [TEST] Reverse-close-shadow failed for {symbol}: {e}")
                    return {'status': 'rejected', 'reason': f'[TEST] reverse-close failed: {e}'}
                res = self._open_shadow(symbol, side, entry_price, opened_by) or {}
                if res.get('ok'):
                    log_activity(symbol, 'opened', f'Реверс (paper): {existing_shadow["side"]} → {side}', side=side, source='TM')
                    return {'shadow_opened': True, 'status': 'opened', 'is_paper': True}
                log_activity(symbol, 'rejected', f'Реверс-відкриття (paper) відхилено: {res.get("reason", "blocked")}', side=side, source='TM')
                return {'shadow_opened': False, 'status': 'rejected', 'is_paper': True,
                        'reason': res.get('reason', 'shadow reverse-open blocked')}
            res = self._open_shadow(symbol, side, entry_price, opened_by) or {}
            if res.get('ok'):
                log_activity(symbol, 'opened', 'Пряме відкриття (paper) — черги не перехопили сигнал', side=side, source='TM')
                return {'shadow_opened': True, 'status': 'opened', 'is_paper': True}
            log_activity(symbol, 'rejected', f'Paper-відкриття відхилено: {res.get("reason", "shadow open blocked")}', side=side, source='TM')
            return {'shadow_opened': False, 'status': 'rejected', 'is_paper': True,
                    'reason': res.get('reason', 'shadow open blocked')}

        # Neither track acted — report why so the marker isn't "unknown".
        if not enabled and not test_mode:
            real_reason = real_reason or 'TM is disabled and Test Mode is off — nothing to open'
        log_activity(symbol, 'skipped', real_reason or 'сигнал не оброблено (жоден трек не активний)', side=side, source='TM')
        return {'real_opened': False, 'shadow_opened': False, 'status': 'rejected',
                'reason': real_reason or 'signal not actioned'}
    
    def on_choch_event(self, symbol: str, direction: str, level: float, bar_t):
        """Called by SMC scanner for EVERY CHoCH detected (regardless of dedup/HTF).
        
        Used to evaluate exit rules:
          - Reverse SMC (CHoCH only): close position when opposite CHoCH appears
          - Forecast 1H Confluence: close when opposite CHoCH AND Forecast 1H also opposite
        
        Both rules work on real (when TM enabled) and shadow (when test_mode on)
        positions. Telegram notifications are sent regardless.
        """
        s = self._settings
        with self._lock:
            real = self._positions.get(symbol)
            shadow = self._shadow_positions.get(symbol)
        
        # Pick whichever position exists (real takes precedence)
        pos = real or shadow
        if not pos:
            return
        
        # === Manual mode gate ===
        # CHoCH-driven exits (Reverse SMC, Forecast 1H Confluence) are
        # AUTOMATIC and therefore bypassed in manual mode. Manual SL/TP
        # and force-close remain the only ways out.
        if pos.get('manual_mode'):
            return
        
        # Record this CHoCH in the evaluator state (whether same-dir or opposite).
        # The evaluator scores opposite CHoCH as a strong negative and same-dir
        # as a small positive — it needs to see ALL CHoCH events after open.
        choch_record = {
            'dir': direction,
            't': float(bar_t) if bar_t else time.time(),
            'level': float(level) if level else None,
        }
        with self._lock:
            if real and symbol in self._pos_state:
                self._pos_state[symbol]['last_choch_after_open'] = choch_record
            if shadow and symbol in self._shadow_pos_state:
                self._shadow_pos_state[symbol]['last_choch_after_open'] = choch_record
        
        pos_dir = 'bull' if pos['side'] == 'LONG' else 'bear'
        if direction == pos_dir:
            return  # same-direction CHoCH — not a reversal

        # 🔄 «Реверс через Чергу-2»: when ON, a bare opposite CHoCH must NOT close
        # the position here — the reversal is owned by Queue 2, which closes the
        # opposite trade only AFTER the opposite signal fully passes its algorithm
        # (SCORE+CTR). Otherwise this raw-CHoCH close would bypass Queue 2.
        try:
            from detection.fuel_filter import get_fuel_filter
            _ffx = get_fuel_filter()
            if _ffx and _ffx.is_enabled():
                _fs = _ffx.get_settings()
                if _fs.get('queue2_enabled') and _fs.get('queue2_reverse_via_queue', True):
                    return
        except Exception:
            pass

        # === Forecast 1H Confluence Close ===
        # Highest priority: if both LTF reverse AND Forecast 1H opposite, close.
        if s.get('use_forecast_1h_close'):
            forecast = self._get_forecast_1h(symbol)
            if forecast and forecast.get('side') in (1, -1):
                forecast_dir = 'bull' if forecast['side'] == 1 else 'bear'
                if forecast_dir == direction:
                    # Confluence! Forecast and CHoCH both oppose position
                    current_price = self._get_current_price(symbol) or pos['entry_price']
                    if real:
                        self._close_position(symbol, current_price,
                                              reason='forecast_1h_confluence')
                    elif shadow:
                        self._close_shadow(symbol, current_price,
                                            reason='forecast_1h_confluence')
                    return  # closed; don't fall through to plain reverse

        # === Plain Reverse SMC (CHoCH only) ===
        if s.get('use_reverse_smc'):
            current_price = self._get_current_price(symbol) or pos['entry_price']
            if real:
                self._close_position(symbol, current_price, reason='reverse_smc')
            elif shadow:
                self._close_shadow(symbol, current_price, reason='reverse_smc')
    
    def on_main_ob_update(self, symbol: str, ob_data: Optional[Dict] = None):
        """Called by SMC scanner whenever any OB might have changed for `symbol`.
        
        TM reads its CONFIGURED `opposite_ob_exit_timeframe` from DB and
        evaluates the rule against that. The `ob_data` parameter is kept
        for backward compatibility but is no longer the source of truth —
        the configured TF may differ from whatever ob_data the scanner
        passes (e.g. user picked 1h exit but scanner passed 15m main-TF
        OB). DB read ensures we always check the right TF.
        
        Implements the "opposite-OB exit" rule: if a position is open and
        the OB on the configured TF has the OPPOSITE bias to the position's
        direction, close immediately. Triggered by bias-non-None — bias=None
        means "no valid OB right now" and is ignored (no signal to act on).
        
        Why hooked here vs in _tick: shadow positions aren't monitored on
        price ticks (they're event-driven), and the OB bias only changes
        when the scanner re-runs the detector. Calling this from inside
        the scanner's per-symbol pass means we react in the same pass
        that computed the new OB — no polling, no race window.
        
        Gate is the global `use_opposite_ob_exit` toggle. Off by default;
        users who want this behavior opt in via the Exit Rules block.
        """
        s = self._settings
        if not s.get('use_opposite_ob_exit'):
            return
        
        with self._lock:
            real = self._positions.get(symbol)
            shadow = self._shadow_positions.get(symbol)
        
        pos = real or shadow
        if not pos:
            return  # No open position for this symbol
        
        # Read OB at the configured exit TF from DB.
        # We don't trust ob_data parameter — scanner may have passed a
        # different TF's OB. The DB row at our configured TF is authoritative.
        exit_tf = s.get('opposite_ob_exit_timeframe', '15m')
        try:
            from storage.db_operations import get_db
            row = get_db().get_smc_ob_state(symbol, exit_tf)
        except Exception as e:
            print(f"[TM] opposite_ob_exit DB read error for {symbol}@{exit_tf}: {e}")
            return
        
        if row is None or not row.get('bias'):
            # Either scanner never computed this TF, or no valid OB right now.
            # Don't treat absence as opposite signal; OBs come and go on every
            # mitigation, and we'd churn closes.
            return
        
        ob_bias = row['bias']
        pos_long = pos['side'] == 'LONG'
        
        # OB is opposite when:
        #   position LONG  and OB BEARISH   → close
        #   position SHORT and OB BULLISH   → close
        is_opposite = ((pos_long and ob_bias == 'BEARISH')
                       or (not pos_long and ob_bias == 'BULLISH'))
        if not is_opposite:
            return  # Same-side OB — keep position; OB is supporting it
        
        # === Close ===
        current_price = self._get_current_price(symbol) or pos['entry_price']
        ob_tag = (row.get('created_by_tag') or '').upper()
        # Diagnostic log so user can correlate exits with OB events
        print(f"[TM] 🔃 Opposite OB exit triggered for {symbol} "
              f"({pos['side']} → opposite OB={ob_bias}/{ob_tag or '?'} "
              f"@ {exit_tf})")
        
        if real:
            self._close_position(symbol, current_price,
                                  reason='opposite_ob_exit')
        elif shadow:
            self._close_shadow(symbol, current_price,
                                reason='opposite_ob_exit')
    
    def _get_forecast_1h(self, symbol: str) -> Optional[Dict]:
        """Read the latest Forecast 1H for the symbol from forecast_engine cache."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return None
            cached = fe.get(symbol)
            if not cached:
                return None
            return cached.get('forecast_1h')
        except Exception:
            return None
    
    def _get_forecast_both(self, symbol: str) -> Optional[Dict]:
        """Read both 1H and 4H forecasts, returned in the dual-TF shape
        {f1_side, f1_conf, f4_side, f4_conf} consumed by the evaluator's
        two independent forecast levers. Also carries legacy 'side'/'confidence'
        (= 1H) so any older reader keeps working."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return None
            cached = fe.get(symbol)
            if not cached:
                return None
            f1 = cached.get('forecast_1h') or {}
            f4 = cached.get('forecast_4h') or {}
            return {
                'f1_side': f1.get('side', 0), 'f1_conf': f1.get('confidence', 0),
                'f4_side': f4.get('side', 0), 'f4_conf': f4.get('confidence', 0),
                # legacy 1H fields for backward compat
                'side': f1.get('side', 0), 'confidence': f1.get('confidence', 0),
            }
        except Exception:
            return None
    
    def _get_ctr(self, symbol: str) -> Optional[Dict]:
        """Read the latest CTR (STC) for the symbol from forecast_engine cache."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return None
            cached = fe.get(symbol)
            if not cached:
                return None
            return cached.get('ctr')
        except Exception:
            return None
    
    def _format_opened_by(self, opened_by: str, symbol: str,
                          entry_score: Optional[Dict] = None) -> str:
        """Build a compact opened_by label.
        
        Format: '<base> · 🤪 LONG 78% (good)'
        
        Previously this surfaced Forecast/CTR/Entry as three separate
        chunks but the user feedback was that it was hard to read. Now
        we delegate the whole verdict to the Decision Center headline,
        which is one phrase humans actually parse at a glance. The base
        opened_by tag (e.g. 'choch_bos', 'choch') is preserved as the
        prefix so existing logic that inspects it keeps working.
        
        Falls back to '<base>' alone when no decision is available.
        """
        parts = [opened_by]
        # FF auto-engine trades label "Opened by" with their own tag (🔥 entry
        # exhaustion, or legacy 🕯️) — keep it as-is, no 🤪 verdict suffix.
        if opened_by and (str(opened_by).startswith('🔥')
                          or str(opened_by).startswith('🕯️')):
            return opened_by
        if entry_score:
            headline = entry_score.get('headline')
            verdict = self._verdict_ua(entry_score.get('verdict', ''))
            if headline:
                parts.append(f"🤪 {headline} ({verdict})" if verdict else f"🤪 {headline}")
            else:
                # Legacy entry_score shape — score+verdict only
                score = entry_score.get('score')
                if score is not None:
                    sign = '+' if score >= 0 else ''
                    if verdict:
                        parts.append(f"🤪 {sign}{score:.0f} ({verdict})")
                    else:
                        parts.append(f"🤪 {sign}{score:.0f}")
        return ' · '.join(parts)
    
    def _base_opened_by(self, opened_by: str) -> str:
        """Strip the contextual suffix added by _format_opened_by.
        Returns just the base tag ('choch_bos', 'choch', or 'manual').
        Used for any logic that needs to compare opened_by tags.
        """
        if not opened_by:
            return ''
        return opened_by.split(' · ', 1)[0].strip()
    
    @staticmethod
    def _fresh_pos_state() -> Dict:
        """Build a blank evaluator-state record. One per open position."""
        return {
            'peak_pnl_pct': 0.0,
            'bos_with_count': 0,
            'bos_against_count': 0,
            'last_choch_after_open': None,  # {dir, t, level} or None
        }

    @staticmethod
    def _history_peak(pos: Dict) -> float:
        """Best PnL% the trade already reached, recovered from the position's
        persisted field + recorded history samples. Used to seed peak_pnl_pct on
        restart so the soft trailing-TP survives a bot restart (Render sleep)."""
        cands = [0.0]
        pp = (pos or {}).get('peak_pnl_pct')
        if pp is not None:
            try:
                cands.append(float(pp))
            except (TypeError, ValueError):
                pass
        for h in (pos or {}).get('history') or []:
            hp = h.get('pnl')
            if hp is not None:
                try:
                    cands.append(float(hp))
                except (TypeError, ValueError):
                    pass
        return round(max(cands), 4)
    
    def _build_eval_config(self):
        """Construct an EvaluationConfig from current settings.
        
        Resolution order for threshold:
          1) explicit override field health_score_threshold_override
          2) value from selected preset
          3) default (Balanced)
        Weights come from settings.health_score_weights, with any missing
        fields backed by the EvaluationConfig defaults.
        """
        from detection.position_evaluator import EvaluationConfig, PRESETS
        s = self._settings
        weights = dict(s.get('health_score_weights') or {})
        
        # Resolve threshold
        override = s.get('health_score_threshold_override')
        if override is not None:
            try:
                threshold = float(override)
            except Exception:
                threshold = None
        else:
            threshold = None
        if threshold is None:
            preset = s.get('health_score_preset', 'balanced')
            preset_cfg = PRESETS.get(preset, PRESETS['balanced'])
            threshold = preset_cfg.get('threshold', -40.0)
        
        weights['threshold'] = float(threshold)
        return EvaluationConfig.from_dict(weights)
    
    def _build_entry_eval_config(self):
        """Mirror of _build_eval_config for the Entry Score side."""
        from detection.position_evaluator import EntryEvaluationConfig, ENTRY_PRESETS
        s = self._settings
        weights = dict(s.get('entry_score_weights') or {})
        
        override = s.get('entry_score_threshold_override')
        if override is not None:
            try:
                threshold = float(override)
            except Exception:
                threshold = None
        else:
            threshold = None
        if threshold is None:
            preset = s.get('entry_score_preset', 'balanced')
            preset_cfg = ENTRY_PRESETS.get(preset, ENTRY_PRESETS['balanced'])
            threshold = preset_cfg.get('threshold', 30.0)
        
        weights['threshold'] = float(threshold)
        return EntryEvaluationConfig.from_dict(weights)
    
    def _gather_entry_context(self, symbol: str, side: str,
                               entry_price: float) -> Dict:
        """Pull every datum the entry evaluator needs from scanner caches.
        
        This is intentionally best-effort — any field that fails to resolve
        is left as None and the corresponding scorer returns 0. The evaluator
        gracefully degrades; we don't want a missing ATR to crash the open.
        
        Returns a dict of kwargs ready to pass to evaluate_entry().
        """
        ctx = {
            'side': side,
            'entry_price': entry_price,
            'htf_bias': 'neutral',
            'forecast': None, 'ctr': None,
            'choch_age_bars': None,
            'strong_low': None, 'weak_high': None,
            'range_high': None, 'range_low': None,
            'atr': None,
            'signal_volume': None, 'avg_volume': None,
        }
        
        if not self.scanner:
            return ctx
        
        # HTF bias
        try:
            ctx['htf_bias'] = self.scanner._htf_cache.get(symbol, {}).get('bias', 'neutral')
        except Exception:
            pass
        
        # Forecast 1H + CTR (already cached by forecast_engine)
        ctx['forecast'] = self._get_forecast_both(symbol)
        ctx['ctr'] = self._get_ctr(symbol)
        
        # Klines cache for ATR / volume / pivot extraction
        try:
            with self.scanner._lock:
                cached = dict(self.scanner._cache.get(symbol) or {})
        except Exception:
            cached = {}
        
        klines = cached.get('klines') or []
        analysis = cached.get('analysis') or {}
        
        # ---- ATR (period 14, latest bar) ----
        if klines and len(klines) >= 15:
            try:
                from detection.forecast_engine import _atr
                atr_series = _atr(klines, 14)
                if atr_series:
                    ctx['atr'] = atr_series[-1]
            except Exception:
                pass
        
        # ---- Volume confirmation ----
        if klines and len(klines) >= 21:
            try:
                # Signal bar: the most recent CLOSED bar (klines[-2] is safer
                # than klines[-1] which may still be in-progress, but we want
                # the bar that triggered the signal — typically the last
                # closed bar i.e. klines[-2]).
                signal_bar = klines[-2] if len(klines) >= 2 else klines[-1]
                ctx['signal_volume'] = float(signal_bar.get('v', 0) or 0)
                # Average volume — prior 20 closed bars, excluding the signal bar
                avg_window = klines[-22:-2] if len(klines) >= 22 else klines[:-2]
                vols = [float(k.get('v', 0) or 0) for k in avg_window]
                vols = [v for v in vols if v > 0]
                if vols:
                    ctx['avg_volume'] = sum(vols) / len(vols)
            except Exception:
                pass
        
        # ---- CHoCH age + Strong Low / Weak High + range bounds ----
        # The analysis dict comes from detect_smc_structure which exposes
        # 'pivots' (list with idx, price, type ∈ HH/HL/LH/LL),
        # 'last_choch' (most recent CHoCH event with idx).
        try:
            pivots = analysis.get('pivots') or []
            last_choch = analysis.get('last_choch')
            
            # CHoCH age in bars (klines length minus event index)
            if last_choch and 'idx' in last_choch and klines:
                ctx['choch_age_bars'] = max(0, len(klines) - 1 - int(last_choch['idx']))
            
            # Strong Low = most recent HL (in bullish context)
            # Weak High = most recent LH (in bearish context)
            # We walk pivots backward to find each
            for p in reversed(pivots):
                if ctx['strong_low'] is None and p.get('type') == 'HL':
                    ctx['strong_low'] = float(p.get('price', 0))
                if ctx['weak_high'] is None and p.get('type') == 'LH':
                    ctx['weak_high'] = float(p.get('price', 0))
                if ctx['strong_low'] is not None and ctx['weak_high'] is not None:
                    break
            
            # Range — high/low of last 20 bars as a robust proxy.
            # (Pine PD zone uses swing high/low; for our advisory score this
            # short-window range is good enough and always available.)
            if klines and len(klines) >= 20:
                window = klines[-20:]
                ctx['range_high'] = max(float(k.get('h', k.get('p', 0))) for k in window)
                ctx['range_low'] = min(float(k.get('l', k.get('p', 0))) for k in window)
        except Exception:
            pass
        
        return ctx
    
    def _compute_entry_score(self, symbol: str, side: str,
                              entry_price: float) -> Optional[Dict]:
        """Compute the Entry Score for a fresh signal. Best-effort, never
        raises. Returns None when the feature is disabled or evaluation fails.
        """
        if not self._settings.get('entry_score_enabled', True):
            return None
        try:
            from detection.position_evaluator import evaluate_entry
        except Exception:
            return None
        
        try:
            ctx = self._gather_entry_context(symbol, side, entry_price)
            cfg = self._build_entry_eval_config()
            return evaluate_entry(config=cfg, **ctx)
        except Exception as e:
            print(f"[TM] Entry score error for {symbol}: {e}")
            return None
    
    def compute_decision(self, symbol: str,
                          current_price: float) -> Optional[Dict]:
        """The single source of truth for "what does the bot think about
        this symbol right now?". Returns a Decision Center verdict ready
        to render in UI / quote in Telegram / log to history.
        
        This replaces the older split between compute_directional_bias()
        (Entry-side) and chart_health snapshots (Health-side). The frontend
        renders one block from this output.
        
        Returns None when the feature is disabled. Never raises.
        """
        if not self._settings.get('entry_score_enabled', True):
            return None
        
        try:
            from detection.position_evaluator import evaluate_entry
            from detection import decision_center
        except Exception:
            return None
        
        try:
            cfg = self._build_entry_eval_config()
            ctx_long = self._gather_entry_context(symbol, 'LONG', current_price)
            ctx_short = self._gather_entry_context(symbol, 'SHORT', current_price)
            res_long = evaluate_entry(config=cfg, **ctx_long)
            res_short = evaluate_entry(config=cfg, **ctx_short)
        except Exception as e:
            print(f"[TM] compute_decision evaluator error for {symbol}: {e}")
            return None
        
        # Position context — provide health/PnL if a position exists on this
        # symbol. Real takes precedence over shadow.
        position = None
        try:
            with self._lock:
                real_pos = dict(self._positions.get(symbol) or {})
                shadow_pos = dict(self._shadow_positions.get(symbol) or {})
            
            for pos, kind, is_shadow in [
                (real_pos, 'real', False),
                (shadow_pos, 'shadow', True),
            ]:
                if not pos:
                    continue
                # Compute live PnL
                entry = pos.get('entry_price', 0)
                pnl_pct = 0.0
                if entry > 0:
                    if pos['side'] == 'LONG':
                        pnl_pct = (current_price - entry) / entry * 100
                    else:
                        pnl_pct = (entry - current_price) / entry * 100
                health = self._compute_health(pos, is_shadow=is_shadow)
                position = {
                    'kind': kind,
                    'side': pos['side'],
                    'entry_price': entry,
                    'pnl_pct': round(pnl_pct, 3),
                    'health': health,
                }
                break  # real first, only one position per symbol expected
        except Exception:
            pass
        
        # Pull market state from contexts (already gathered)
        return decision_center.build_decision(
            long_eval=res_long,
            short_eval=res_short,
            htf_bias=ctx_long.get('htf_bias'),
            forecast=ctx_long.get('forecast'),
            ctr=ctx_long.get('ctr'),
            position=position,
        )
    
    # Legacy alias kept for backward-compat — the older chart_data response
    # exposed `directional_bias`. Frontends gradually migrate to `decision`.
    def compute_directional_bias(self, symbol: str,
                                  current_price: float) -> Optional[Dict]:
        return self.compute_decision(symbol, current_price)
    
    def _compute_health(self, pos: Dict, is_shadow: bool) -> Optional[Dict]:
        """Run the position evaluator for a single open position. Pulls
        fresh values from the SMC scanner's HTF cache and the forecast
        engine, plus the per-position state we maintain in _pos_state.
        
        Returns the evaluator's result dict, or None if Health Score is
        disabled / not enough data to compute meaningfully.
        """
        if not self._settings.get('health_score_enabled', True):
            return None
        try:
            from detection.position_evaluator import evaluate_position
        except Exception:
            return None
        
        symbol = pos.get('symbol')
        if not symbol:
            return None
        
        # Pull market state
        htf_bias = 'neutral'
        if self.scanner:
            try:
                htf_bias = self.scanner._htf_cache.get(symbol, {}).get('bias', 'neutral')
            except Exception:
                pass
        
        forecast = self._get_forecast_1h(symbol)
        ctr = self._get_ctr(symbol)
        
        # Per-position state (keyed by symbol in the right dict)
        state_dict = self._shadow_pos_state if is_shadow else self._pos_state
        state = state_dict.get(symbol) or self._fresh_pos_state()
        
        # Current PnL
        current = self._get_current_price(symbol) or pos['entry_price']
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (current - entry) / entry * 100
        else:
            pnl_pct = (entry - current) / entry * 100
        
        # Update peak inline (so the evaluator always sees latest peak even
        # if the monitor loop hasn't ticked since the last price move)
        if pnl_pct > state.get('peak_pnl_pct', 0):
            state['peak_pnl_pct'] = pnl_pct
        
        try:
            cfg = self._build_eval_config()
            return evaluate_position(
                side=pos['side'],
                pnl_pct=pnl_pct,
                opened_at=pos.get('opened_at', time.time()),
                htf_bias=htf_bias,
                forecast=forecast,
                ctr=ctr,
                recent_choch=state.get('last_choch_after_open'),
                bos_count_with=state.get('bos_with_count', 0),
                bos_count_against=state.get('bos_against_count', 0),
                peak_pnl_pct=state.get('peak_pnl_pct'),
                config=cfg,
            )
        except Exception as e:
            print(f"[TM] Health eval error for {symbol}: {e}")
            return None
    
    def on_bos_event(self, symbol: str, direction: str, level: float, bar_t: int):
        """Called by SMC scanner when a BOS event is detected.
        
        Three responsibilities:
          1) Update Health Score evaluator state (BOS counts in BOTH directions)
             for real AND shadow positions, always — regardless of TM enabled.
          2) Maintain `bos_count_since_entry` on positions so we can
             reference "BOS #2", "BOS #3", etc. The opening BOS (when
             opened_by=choch_bos) counts as #1; opening on `choch` mode
             starts the same way because the user explicitly mapped both
             modes to BOS-2 = first BOS after entry.
          3) Trigger BOS-N partial closes — for both REAL (via Bybit API)
             and SHADOW (paper trading book-keeping) positions, gated by
             use_bos_partials.
        
        Idempotency: scanner may emit the same BOS event multiple times
        across re-scans of the same bar, especially around re-deploys or
        cache invalidations. We dedupe on (symbol, position-instance, bar_t)
        so a single physical BOS bar increments the counter exactly once
        per position. Position-instance is implicit: when a new position
        opens we reset `last_bos_bar_t` to None.
        
        Diagnostic logging is intentionally verbose so production logs
        on Render show exactly what happened on every BOS — without it,
        debugging "BOS-2 didn't fire" is essentially blind.
        """
        s = self._settings
        with self._lock:
            real = self._positions.get(symbol)
            shadow = self._shadow_positions.get(symbol)
            
            # === Evaluator-state updates (always, both kinds) ===
            # These also need bar_t-based dedup so health scores aren't
            # inflated when scanner re-emits the same event.
            for pos, state_dict, kind_label in (
                (real, self._pos_state, 'real'),
                (shadow, self._shadow_pos_state, 'shadow'),
            ):
                if not pos or symbol not in state_dict:
                    continue
                # Per-position last-seen BOS timestamp — separate from the
                # per-position partial counter so evaluator state is updated
                # even if partial close was already done at this bar.
                last_t = state_dict[symbol].get('last_bos_bar_t')
                if last_t is not None and bar_t and last_t >= bar_t:
                    # Same or older bar — already counted, skip
                    continue
                state_dict[symbol]['last_bos_bar_t'] = bar_t
                
                pos_dir = 'bull' if pos['side'] == 'LONG' else 'bear'
                if direction == pos_dir:
                    state_dict[symbol]['bos_with_count'] = (
                        state_dict[symbol].get('bos_with_count', 0) + 1)
                else:
                    state_dict[symbol]['bos_against_count'] = (
                        state_dict[symbol].get('bos_against_count', 0) + 1)
        
        # === Partial-close logic ===
        # Two independent code paths (both gated by use_bos_partials only):
        #   1) Real positions — operate on _positions, call Bybit via _partial_close
        #   2) Shadow positions — operate on _shadow_positions, no Bybit calls.
        # Neither is gated by the enabled toggle: the toggle only blocks NEW
        # entries. An already-open position is always worked through.
        # Both paths run on every BOS event so a symbol with both a real
        # AND a shadow position would partial-close both. In practice
        # users only have one of the two at a time but the code is robust
        # to either configuration.

        if not s.get('use_bos_partials'):
            # User explicitly disabled the feature — log once and skip both paths.
            print(f"[TM] BOS {symbol} {direction} ignored: use_bos_partials=False")
            return

        # ----- REAL position path -----
        if real:
            # Manual mode skips auto partial-closes; evaluator state above
            # still updated so reverting manual mode works seamlessly.
            if real.get('manual_mode'):
                print(f"[TM] BOS {symbol} {direction} for real {symbol}: "
                      f"partial-close skipped (manual mode)")
            else:
                self._process_bos_real(symbol, direction, level, bar_t, real)

        # ----- SHADOW position path -----
        # Always runs when a shadow position exists. test_mode by design
        # operates while TM master toggle is off.
        if shadow:
            if shadow.get('manual_mode'):
                print(f"[TM] BOS {symbol} {direction} for shadow {symbol}: "
                      f"partial-close skipped (manual mode)")
            else:
                self._process_bos_shadow(symbol, direction, level, bar_t)
        elif not real:
            # Neither real nor shadow — nothing to do. Don't log to avoid
            # noise from BOS events on watched symbols without trades.
            pass
    
    def _process_bos_real(self, symbol: str, direction: str, level: float,
                          bar_t: int, real: Dict):
        """BOS-N partial close handler for the REAL position on `symbol`.
        Extracted from on_bos_event so the two code paths (real vs shadow)
        stay readable at the top level.
        
        Idempotent on bar_t — the same physical BOS bar will not increment
        the counter twice even if scanner re-emits.
        """
        s = self._settings
        
        # Same direction as our position?
        bos_side = 'LONG' if direction == 'bull' else 'SHORT'
        if bos_side != real['side']:
            print(f"[TM] BOS {symbol} {direction} ignored: opposite to position {real['side']}")
            return
        
        # bar_t-based dedup. If we've already processed a BOS at this bar
        # for this position, skip silently. Different bar_t (newer BOS) is
        # always processed.
        last_t = real.get('last_bos_bar_t')
        if last_t is not None and bar_t and last_t >= bar_t:
            return
        real['last_bos_bar_t'] = bar_t
        
        # Increment BOS counter. The starting value of 1 (set in
        # _new_position) means the FIRST post-entry BOS becomes #2 — which
        # the user has confirmed matches their settings layout where
        # bos_2_close_pct = "first BOS after entry" for both alert modes.
        prev_n = real.get('bos_count_since_entry', 1)
        real['bos_count_since_entry'] = prev_n + 1
        n = real['bos_count_since_entry']
        
        # Look up the configured percentage for this BOS number
        partial_pct = s.get(f'bos_{n}_close_pct', 0)
        marker_key = f'bos_{n}'
        already_done = marker_key in real.get('partial_closes_done', [])
        
        print(f"[TM] BOS-{n} for {symbol} {real['side']}: "
              f"pct={partial_pct}%, done={already_done}, "
              f"opened_by={self._base_opened_by(real.get('opened_by', ''))}")
        
        if partial_pct <= 0:
            # User has BOS-N partial disabled (e.g. bos_3_close_pct=0)
            return
        if already_done:
            # Idempotency guard — same BOS event delivered twice shouldn't
            # double-close. Can happen on scanner restarts or duplicate hooks.
            return
        
        current_price = self._get_current_price(symbol) or level
        try:
            self._partial_close(symbol, partial_pct, current_price,
                                reason=f'bos_{n}_partial')
            real.setdefault('partial_closes_done', []).append(marker_key)
            print(f"[TM] ✂️ BOS-{n} partial close executed: "
                  f"{symbol} {real['side']} {partial_pct}% @ {self._fmt_price(current_price)}")
        except Exception as e:
            print(f"[TM] ❌ BOS-{n} partial close FAILED for {symbol}: {e}")
            return
        
        # Auto-activate trailing after BOS-2 — this works even when the
        # global use_trailing toggle is OFF, because the user explicitly
        # opted into "trailing after BOS-2" as a separate gate. Note that
        # _update_trailing checks use_trailing before tracking — so we
        # also need to bypass that gate at update time. See _update_trailing
        # for the matching logic that respects the BOS-activated state.
        if n == 2 and s.get('trailing_after_bos_2'):
            real['trailing_active'] = True
            real['trailing_peak'] = current_price
            real['trailing_via_bos2'] = True  # marker — bypass use_trailing gate
            self._notify(f"📈 Trailing активовано після BOS-2: {symbol}")
            print(f"[TM] 📈 Trailing activated for {symbol} via BOS-2 hook "
                  f"(use_trailing={s.get('use_trailing')})")
        
        self._persist_positions()
    
    def _process_bos_shadow(self, symbol: str, direction: str, level: float, bar_t: int):
        """Mirror of the BOS-N partial-close logic for SHADOW (paper) positions.
        
        Called from on_bos_event right after the real-position handling.
        Same gates (use_bos_partials), same counter logic (first BOS after
        entry → bos_2_close_pct), same trailing-after-BOS-2 wiring. The only
        differences:
          - Calls _partial_close_shadow (no Bybit API)
          - Doesn't require TM master toggle to be enabled (test mode is
            specifically designed to run while the master toggle is OFF)
          - Idempotency uses partial_closes_done list inside the shadow pos
        
        This is what makes "BOS-N partials in test mode" actually work.
        Without it, BOS events were tracked in evaluator state but never
        triggered any shadow-side partial closes.
        """
        s = self._settings
        if not s.get('use_bos_partials'):
            return  # logged once at the top-level on_bos_event
        
        with self._lock:
            shadow = self._shadow_positions.get(symbol)
        if not shadow:
            return
        
        # Same direction as our position?
        bos_side = 'LONG' if direction == 'bull' else 'SHORT'
        if bos_side != shadow['side']:
            print(f"[TM] [TEST] BOS {symbol} {direction} ignored: "
                  f"opposite to shadow position {shadow['side']}")
            return
        
        # bar_t-based dedup so re-emitted BOS for the same bar doesn't
        # over-increment the counter. Mirrors the real-side guard.
        last_t = shadow.get('last_bos_bar_t')
        if last_t is not None and bar_t and last_t >= bar_t:
            return
        shadow['last_bos_bar_t'] = bar_t
        
        # Increment counter (mirrors real-side semantics)
        prev_n = shadow.get('bos_count_since_entry', 1)
        shadow['bos_count_since_entry'] = prev_n + 1
        n = shadow['bos_count_since_entry']
        
        partial_pct = s.get(f'bos_{n}_close_pct', 0)
        marker_key = f'bos_{n}'
        already_done = marker_key in shadow.get('partial_closes_done', [])
        
        print(f"[TM] [TEST] BOS-{n} for {symbol} {shadow['side']} (shadow): "
              f"pct={partial_pct}%, done={already_done}")
        
        if partial_pct <= 0:
            return
        if already_done:
            return
        
        current_price = self._get_current_price(symbol) or level
        try:
            self._partial_close_shadow(symbol, partial_pct, current_price,
                                        reason=f'bos_{n}_partial')
            # Note: _partial_close_shadow may have already removed the
            # shadow position if remaining qty hit zero. Re-fetch before
            # marking partial_closes_done.
            with self._lock:
                still_open = self._shadow_positions.get(symbol)
            if still_open:
                still_open.setdefault('partial_closes_done', []).append(marker_key)
                self._persist_shadow_positions()
        except Exception as e:
            print(f"[TM] ❌ [TEST] BOS-{n} shadow partial FAILED for {symbol}: {e}")
            return
        
        # Auto-activate trailing for shadow on BOS-2 (same logic as real).
        # Note: shadow positions don't go through _update_trailing yet —
        # there's no shadow monitor loop. We set the markers anyway so that
        # if a future shadow monitor is added, the state is already there.
        if n == 2 and s.get('trailing_after_bos_2'):
            with self._lock:
                still_open = self._shadow_positions.get(symbol)
            if still_open:
                still_open['trailing_active'] = True
                still_open['trailing_peak'] = current_price
                still_open['trailing_via_bos2'] = True
                self._persist_shadow_positions()
                self._notify(
                    f"📈 [TEST] Trailing активовано після BOS-2: {symbol}",
                    is_test=True,
                )
                print(f"[TM] [TEST] 📈 Shadow trailing activated for {symbol} via BOS-2 hook")
    
    # ============================================================
    # Position open / close / partial
    # ============================================================
    
    def _archive_closed(self, closed: Dict, pos: Dict, is_paper: bool):
        """Append a closed trade to the permanent DB archive (never trimmed).
        Carries the pre-trade snapshot captured at open. Guarded so it can
        never disrupt the close flow. Only archives if the archive toggle is ON."""
        if not self.db:
            return
        # Check if archiving is enabled
        settings = self.get_settings()
        if not settings.get('archive_trades', False):
            return
        try:
            snap = pos.get('entry_snapshot')
            self.db.archive_trade(closed, entry_snapshot=snap, is_paper=is_paper)
        except Exception as e:
            print(f"[TM] archive_closed error: {e}")

    def _capture_entry_snapshot(self, symbol: str, side: str,
                                entry_price: float, decision) -> Dict:
        """Snapshot the full pre-trade analysis at OPEN time. This is the
        feature set for backtesting entry quality later. Pure capture —
        never affects whether the trade opens."""
        snap = {
            'ts': time.time(),
            'side': side,
            'entry_price': entry_price,
        }
        # Entry decision (directional conviction) — compact form
        if decision:
            snap['decision'] = {
                'verdict': decision.get('verdict'),
                'recommended': decision.get('recommended'),
                'confidence': decision.get('confidence'),
                'prob_long': decision.get('prob_long'),
                'prob_short': decision.get('prob_short'),
                'headline': decision.get('headline'),
            }
        # Move potential (ATR / runway / exhaustion)
        try:
            from detection.move_potential import analyze_move_potential
            klines = None
            if self.scanner is not None:
                getc = getattr(self.scanner, '_get_cached_klines', None)
                if callable(getc):
                    klines = getc(symbol)
            if klines and len(klines) >= 20:
                mv = analyze_move_potential(side=side, klines=klines)
                if mv.get('ok'):
                    snap['move'] = {
                        'atr_pct': mv.get('atr_pct'),
                        'stretch_atr': mv.get('stretch_atr'),
                        'runway_pct': mv.get('runway_pct'),
                        'runway_atr': mv.get('runway_atr'),
                        'adr_used_pct': mv.get('adr_used_pct'),
                        'exhaustion': mv.get('exhaustion'),
                        'verdict': mv.get('verdict'),
                    }
        except Exception as e:
            print(f"[TM] snapshot move error: {e}")
        # Hold score at open
        try:
            hold = self._compute_hold_score({'symbol': symbol, 'side': side,
                                             'entry_price': entry_price,
                                             'current_price': entry_price})
            if hold and hold.get('ok'):
                snap['hold'] = {'score': hold.get('score'),
                                'verdict': hold.get('verdict')}
        except Exception as e:
            print(f"[TM] snapshot hold error: {e}")
        return snap

    def _open_position(self, symbol: str, side: str, entry_price: float, opened_by: str, bypass_gates: bool = False):
        """Open a real position. Returns a structured result so callers can
        surface the EXACT reason instead of a generic "order failed":
          {'ok': True}                      — position opened
          {'ok': False, 'reason': <str>}    — blocked/failed, with why

        Signal-flow callers ignore the return; manual_open relays the reason
        to the UI so the user sees what actually happened (gate vs sizing vs
        exchange rejection) rather than "check logs".
        """
        s = self._settings

        # === Global directional gate ===
        # Master toggle ON by default for both sides. When the user flips
        # off LONG (or SHORT), NO real opens go through for that side —
        # regardless of who called us (SMC signal, manual UI, future
        # webhook). Closures and reverses still close existing positions
        # normally; only the open step is blocked.
        # EXCEPT: when bypass_gates=True (Fuel Auto-Filter), always allow.
        if not bypass_gates and not self._side_allowed(side):
            print(f"[TM] 🚫 REAL {side} entries disabled — {symbol} not opened")
            return {'ok': False, 'reason':
                    f'{side} entries are disabled (master {side} toggle is OFF). '
                    f'Enable {side} to open this trade.'}

        # NOTE: the old "Fuel Auto-Filter confirmation gate" (require_fuel_confirm)
        # is retired — the new strategy INTERCEPTS opens up front (on_signal /
        # manual_open queue the coin in FF), so by the time we reach here the
        # open is either FF's own (bypass_gates) or FF is disabled. No pull-gate.

        # Calculate size
        try:
            qty = self._calculate_qty(symbol, entry_price)
        except Exception as e:
            self._notify(f"❌ Sizing error for {symbol}: {e}")
            return {'ok': False, 'reason': f'Position sizing error: {e}'}

        if qty <= 0:
            print(f"[TM] {symbol}: zero quantity calculated, skipping")
            return {'ok': False, 'reason':
                    f'Calculated quantity is 0 for {symbol}. Check Position '
                    f'Sizing (USD amount / leverage) — it may be below the '
                    f'minimum order size for this symbol.'}

        # Calculate SL / TP prices
        sl_price = self._calc_sl_price(side, entry_price, symbol) if s.get('use_sl') else None
        tp_price = self._calc_tp_price(side, entry_price) if s.get('use_tp') else None

        # Set leverage on Bybit
        leverage = s.get('leverage', 10)
        try:
            self.bybit.set_leverage(symbol, leverage)
        except Exception as e:
            print(f"[TM] Leverage set warn for {symbol}: {e}")

        # Place order
        bybit_side = 'Buy' if side == 'LONG' else 'Sell'
        try:
            result = self.bybit.place_order(
                symbol=symbol,
                side=bybit_side,
                qty=qty,
                order_type='Market',
                stop_loss=sl_price,
                take_profit=tp_price,
            )
        except Exception as e:
            self._notify(f"❌ Failed to open {side} {symbol}: {e}")
            return {'ok': False, 'reason': f'Bybit order error: {e}'}

        if not result:
            # Surface the exchange's actual rejection reason when available.
            exch_err = getattr(self.bybit, '_last_order_error', None)
            self._notify(f"❌ Order rejected for {symbol} {side}: {exch_err or 'unknown'}")
            return {'ok': False, 'reason':
                    (f'Bybit rejected the {side} order for {symbol} (qty={qty}): '
                     f'{exch_err}') if exch_err else
                    (f'Bybit rejected the {side} order for {symbol} (qty={qty}). '
                     f'Check API key permissions, margin/balance, and that the '
                     f'symbol is tradeable. See Telegram/logs.')}
        
        order_id = result.get('order_id', '')
        # Compute the unified Decision Center verdict — the same object
        # the chart panel and Telegram messages display. Stored on the
        # position so closed-trade analytics can correlate decision quality
        # with actual outcome.
        decision = None
        try:
            decision = self.compute_decision(symbol, entry_price)
        except Exception as e:
            print(f"[TM] decision compute error on open: {e}")
        opened_by_full = self._format_opened_by(opened_by, symbol,
                                                 entry_score=decision)
        position = _new_position(symbol, side, entry_price, qty,
                                   sl_price, tp_price, order_id, opened_by_full)
        if decision is not None:
            # Stored under entry_score key for backward-compat with the UI
            # column and legacy load_positions code; the dict shape is the
            # Decision Center verdict (headline, recommended, verdict, etc.).
            position['entry_score'] = decision

        # FF SCORE verdict + Exhaustion at open — kept for analytics/back-compat.
        position['ff_score_open'] = self._ff_score_snapshot(symbol)
        position['ff_exh_open'] = self._ff_exhaustion(symbol, side)
        # ММ (fuel) value at open — shown in the Reason detail.
        position['ff_mm_open'] = self._ff_mm_snapshot(symbol)
        # ⚡ CTR entry-gate reading at open (for the хронологія modal).
        position['ctr_open'] = self._ctr_snapshot(symbol)

        # Full pre-trade snapshot for the entry-quality backtest dataset.
        # Captured ONCE at open — decision + move-potential + hold score —
        # so we can later test which signals predicted good vs bad trades.
        try:
            position['entry_snapshot'] = self._capture_entry_snapshot(
                symbol, side, entry_price, decision)
        except Exception as e:
            print(f"[TM] entry snapshot error: {e}")

        with self._lock:
            self._positions[symbol] = position
            # Init evaluator state alongside the position
            self._pos_state[symbol] = self._fresh_pos_state()
        self._persist_positions()

        self._notify_open(position)
        return {'ok': True}

    def _close_externally(self, symbol: str, exit_price: float, reason: str = 'external_close'):
        """Bookkeeping-only close for positions that vanished from Bybit
        without TM's involvement (closed manually, hit a Bybit-side SL/TP,
        or got liquidated). Does NOT call bybit.close_position — the
        position is already gone on the exchange, calling close would error.
        
        Everything else (PnL calc, closed_trades append, persist, notify) is
        the same shape as _close_position so the closed trade looks identical
        to a TM-closed one in stats and the UI.
        """
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return
        
        qty = pos.get('remaining_qty', pos['qty'])
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        pnl_usd = (exit_price - entry) * qty * (1 if pos['side'] == 'LONG' else -1)
        
        closed = {
            'symbol': symbol, 'side': pos['side'],
            'entry_price': entry, 'exit_price': exit_price,
            'qty': pos['qty'],
            'remaining_qty_at_close': qty,
            'opened_at': pos['opened_at'],
            'closed_at': time.time(),
            'pnl_pct': round(pnl_pct, 4),
            'pnl_usd': round(pnl_usd, 2),
            'reason': reason,
            'reason_detail': self._build_reason_detail(
                symbol, pos, reason, round(pnl_pct, 4), is_shadow=False),
            'peak_pnl_pct': self._peak_pnl(symbol, round(pnl_pct, 4), is_shadow=False),
            'exit_decision': self._decision_snapshot(symbol),
            'ctr_open': pos.get('ctr_open'),
            'ctr_close': self._ctr_snapshot(symbol),
            # All Manual SL/TP levels that applied during the trade (+ final) —
            # shown in the trade-history modal.
            'manual_sl_hist': list(pos.get('manual_sl_hist') or []),
            'manual_tp_hist': list(pos.get('manual_tp_hist') or []),
            'manual_sl': pos.get('manual_sl'),
            'manual_tp': pos.get('manual_tp'),
            'opened_by': pos.get('opened_by', ''),
            'partial_closes_done': pos.get('partial_closes_done', []),
            'entry_score': pos.get('entry_score'),
        }
        
        with self._lock:
            self._positions.pop(symbol, None)
            self._pos_state.pop(symbol, None)
            closed['history'] = list(pos.get('history') or [])
            self._closed_trades.append(closed)
            if len(self._closed_trades) > CLOSED_TRADES_LIMIT:
                self._closed_trades = self._closed_trades[-CLOSED_TRADES_LIMIT:]
        self._persist_positions()
        self._persist_closed_trades()
        # Finalize in the background: fetch the exchange's real avg exit price +
        # realized PnL (net of fees), archive, then notify with the corrected
        # numbers. External closes are the worst offenders for stale prices, so
        # the notify is intentionally sent AFTER the truth fetch.
        _side = pos['side']
        def _ext_notify(c):
            _sign = '+' if c.get('pnl_pct', 0) >= 0 else ''
            print(f"[TM] 🔄 External close detected for {symbol}: "
                  f"{_sign}{c.get('pnl_pct', 0):.2f}% (gone from Bybit, reason={reason})")
            self._notify(
                f"🔄 External close: {symbol} {_side}\n"
                f"Entry: {self._fmt_price(c.get('entry_price'))} → "
                f"Exit: {self._fmt_price(c.get('exit_price'))}\n"
                f"PnL: {_sign}{c.get('pnl_pct', 0):.2f}% "
                f"({_sign}${c.get('pnl_usd', 0):.2f})"
            )
        self._finalize_close_async(symbol, closed, _side, pos,
                                   notify_fn=_ext_notify)
    
    def _finalize_close_async(self, symbol: str, closed: Dict, side: str,
                              pos: Dict, notify_fn=None):
        """Run the post-close bookkeeping in a BACKGROUND thread so it never
        delays the actual close (the reduce-only order was already sent) nor
        blocks the monitor loop from checking other positions' SL/TP while we
        wait for Bybit to settle the closed-PnL record.

        Order in the worker: fetch real PnL → archive → notify. The Recent
        Closed Trades table already shows the approximation instantly; this
        updates it to exchange-truth a few seconds later.
        """
        def _run():
            try:
                self._apply_exchange_close_truth(symbol, closed, side)
            except Exception as e:
                print(f"[TM] close-truth async error {symbol}: {e}")
            try:
                self._archive_closed(closed, pos, is_paper=False)
            except Exception as e:
                print(f"[TM] archive async error {symbol}: {e}")
            if notify_fn:
                try:
                    notify_fn(closed)
                except Exception as e:
                    print(f"[TM] close-notify async error {symbol}: {e}")
        threading.Thread(target=_run, daemon=True,
                         name=f"close-finalize-{symbol}").start()

    def _apply_exchange_close_truth(self, symbol: str, closed: Dict,
                                    side: str) -> bool:
        """Overwrite a just-recorded closed trade with the REAL figures from
        Bybit's closed-PnL endpoint: actual avg entry/exit price and realized
        PnL (already net of trading fees + funding). Replaces our own
        last-price-based approximation so the UI/stats match the exchange exactly.

        `closed` is the dict already appended to self._closed_trades — we mutate
        it in place and re-persist. Returns True if real data was applied.

        Bybit settles the closed-PnL record a beat after the reduce-only fill,
        so we retry a few times with a short backoff. On total failure we keep
        our approximation (clearly better than nothing) and flag it.
        """
        if not self.bybit or not hasattr(self.bybit, 'get_closed_pnl'):
            closed['pnl_source'] = 'approx'
            return False

        # The closing order side on Bybit is the OPPOSITE of the position side.
        want_close_side = 'Sell' if side == 'LONG' else 'Buy'
        our_qty = float(closed.get('remaining_qty_at_close')
                        or closed.get('qty') or 0)

        # CRITICAL: only accept a closed-PnL record that belongs to THIS close,
        # i.e. whose Bybit timestamp is at/after the moment we closed. Right
        # after the reduce-only fill Bybit may not have settled the new record
        # yet — if a PREVIOUS trade on this same symbol exists, get_closed_pnl
        # returns that stale record. Grabbing it would overwrite the real PnL
        # with an old trade's numbers (e.g. +2% becomes +0.11%). We use a small
        # negative buffer for clock skew, and keep retrying until the fresh
        # record appears (or fall back to our approximation).
        closed_at = float(closed.get('closed_at') or time.time())
        min_ts_ms = int((closed_at - 30) * 1000)

        record = None
        for attempt in range(5):  # ~0.6 + 1.2 + 1.8 + 2.4 ≈ 6s worst case
            recs = self.bybit.get_closed_pnl(symbol, limit=10)
            if recs:
                # Newest first. Keep only FRESH records (this close, not a
                # previous trade) matching the close side.
                fresh = [r for r in recs
                         if r.get('side') == want_close_side
                         and r.get('avg_exit_price', 0) > 0
                         and int(r.get('updated_time') or r.get('created_time') or 0) >= min_ts_ms]
                if fresh:
                    if our_qty > 0:
                        record = min(
                            fresh, key=lambda r: abs(r.get('qty', 0) - our_qty))
                    else:
                        record = fresh[0]
                    break
            if attempt < 4:
                time.sleep(0.6 * (attempt + 1))

        if not record:
            closed['pnl_source'] = 'approx'
            print(f"[TM] ⚠️ closed-PnL not available for {symbol}; "
                  f"kept approximation")
            return False

        entry = record['avg_entry_price'] or closed.get('entry_price')
        exit_p = record['avg_exit_price']
        pnl_usd = record['closed_pnl']
        if entry and entry > 0:
            if side == 'LONG':
                pnl_pct = (exit_p - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_p) / entry * 100
        else:
            pnl_pct = closed.get('pnl_pct', 0)

        with self._lock:
            closed['entry_price'] = entry
            closed['exit_price'] = exit_p
            closed['pnl_usd'] = round(pnl_usd, 4)
            closed['pnl_pct'] = round(pnl_pct, 4)
            closed['pnl_source'] = 'exchange'
            closed['exchange_order_id'] = record.get('order_id')
        self._persist_closed_trades()
        print(f"[TM] ✅ Exchange truth for {symbol}: entry={entry} "
              f"exit={exit_p} realizedPnL=${pnl_usd:.4f} ({pnl_pct:+.2f}%)")
        return True

    def _adopt_external_position(self, bybit_pos: Dict) -> bool:
        """Adopt a position that exists on Bybit but not in TM's view.

        Happens in three scenarios:
          1. User opened a position MANUALLY on Bybit while TM is running
          2. Position pre-existed from before TM was activated
          3. TM was restarted/redeployed; DB lost the position dict but
             the actual exchange position survived
        
        We treat the live exchange state as the source of truth. The
        adopted position dict mirrors _new_position's shape so all monitor
        logic (HTF flip, Reverse SMC, BOS partials, trailing, BE, manual
        SL/TP, force close) works the same.
        
        IMPORTANT: we DON'T auto-set strategy SL/TP on Bybit for adopted
        positions. The user opened it themselves; their SL/TP (if any)
        is preserved. TM monitors whatever levels Bybit returns. If user
        wants TM-managed SL/TP, they can flip the position into manual
        mode and set Manual SL/Manual TP via the UI.
        
        Returns True if adopted, False if rejected (validation failure).
        """
        symbol = bybit_pos.get('symbol')
        if not symbol:
            return False
        # Bybit returns side as 'Buy'/'Sell'; TM internally uses LONG/SHORT
        bybit_side = bybit_pos.get('side', '')
        if bybit_side == 'Buy':
            side = 'LONG'
        elif bybit_side == 'Sell':
            side = 'SHORT'
        else:
            print(f"[TM] adopt skipped for {symbol}: unknown side '{bybit_side}'")
            return False
        
        entry_price = float(bybit_pos.get('entry_price') or 0)
        qty = float(bybit_pos.get('size') or 0)
        if entry_price <= 0 or qty <= 0:
            print(f"[TM] adopt skipped for {symbol}: invalid entry/qty "
                  f"({entry_price}, {qty})")
            return False
        
        sl_price = bybit_pos.get('stop_loss') or None
        tp_price = bybit_pos.get('take_profit') or None
        # Bybit returns 0.0 for unset SL/TP — normalize to None
        if sl_price is not None and float(sl_price) <= 0:
            sl_price = None
        if tp_price is not None and float(tp_price) <= 0:
            tp_price = None
        
        # Build position dict with opened_by='external' marker.
        # opened_at = now: we don't have real open-time from Bybit's
        # current-positions API. Time-stop will count from adoption, which
        # is the safest behavior (avoids accidental immediate exit).
        pos = _new_position(
            symbol=symbol, side=side,
            entry_price=entry_price, qty=qty,
            sl_price=sl_price, tp_price=tp_price,
            order_id='',                # we don't have the original order id
            opened_by='external',
        )
        # Mark as adopted so UI can show a badge and analytics can
        # distinguish this trade's outcome from TM-initiated trades.
        pos['external'] = True
        pos['ff_score_open'] = self._ff_score_snapshot(symbol)
        pos['ff_mm_open'] = self._ff_mm_snapshot(symbol)
        pos['ctr_open'] = self._ctr_snapshot(symbol)

        with self._lock:
            self._positions[symbol] = pos
            self._pos_state[symbol] = self._fresh_pos_state()
        self._persist_positions()
        
        sl_disp = f"SL=${sl_price}" if sl_price else "no SL"
        tp_disp = f"TP=${tp_price}" if tp_price else "no TP"
        print(f"[TM] 🔄 Adopted external position {symbol} {side} "
              f"@ ${entry_price} qty={qty} ({sl_disp}, {tp_disp})")
        try:
            self._notify(
                f"🔄 Adopted external position\n"
                f"{symbol} {side} @ {self._fmt_price(entry_price)}\n"
                f"qty: {qty} · {sl_disp} · {tp_disp}\n"
                f"TM will now manage exits via its algorithm."
            )
        except Exception:
            pass
        return True
    
    def _reconcile_with_bybit(self) -> Dict:
        """Sync TM's view of open positions with what actually exists on
        Bybit. Two-way reconciliation:
        
          - For each Bybit position not in self._positions → ADOPT it
            (call _adopt_external_position). TM starts managing it.
          - For each self._positions entry not on Bybit → it was closed
            externally; call _close_externally to update bookkeeping.
        
        Runs regardless of the enabled toggle. The toggle only controls
        whether NEW trades are accepted (see on_signal / _open_position).
        Existing positions must always be loaded, synced and managed — when
        disabled we simply stop opening new trades, everything else works as
        usual. Runs only if bybit connector is configured with API key.

        Returns a summary dict for logging/diagnostics.
        """
        if not self.bybit:
            return {'skipped': True, 'reason': 'no bybit connector'}
        try:
            # Use the checked variant so we can tell a real "no positions"
            # apart from an API error / partial page. On error we must NOT
            # treat missing symbols as external closes.
            if hasattr(self.bybit, 'get_positions_checked'):
                live, ok = self.bybit.get_positions_checked()
            else:
                live, ok = (self.bybit.get_positions() or []), True
        except Exception as e:
            print(f"[TM] Reconcile fetch error: {e}")
            return {'error': str(e)}

        if not ok:
            # Exchange returned an error/partial response. Adopting is safe
            # (only adds), but closing on a bad fetch is what caused the
            # phantom external_close storm. Skip the close pass entirely.
            print("[TM] Reconcile: skipped close pass (Bybit fetch not ok)")
        
        live_by_symbol = {p['symbol']: p for p in live if p.get('symbol')}

        # Cache the exchange-truth snapshot so get_state() can render the open
        # positions exactly as they are on Bybit (only when the fetch was clean
        # — a partial page must not wipe the cache to empty).
        if ok:
            self._live_positions = live_by_symbol
            self._live_positions_at = time.time()

        with self._lock:
            tm_symbols = set(self._positions.keys())
        live_symbols = set(live_by_symbol.keys())
        
        adopted = []
        closed_externally = []
        
        # 1. Adopt positions present on Bybit but not in TM
        for sym in live_symbols - tm_symbols:
            if self._adopt_external_position(live_by_symbol[sym]):
                adopted.append(sym)
        
        # 2. Detect external closes: in TM but not on Bybit.
        #    ONLY when the fetch was clean — a partial/error page must never
        #    trigger closes (root cause of the phantom-close storm).
        to_close = tm_symbols - live_symbols
        # Sanity guard: if a single reconcile wants to close a large share of
        # all tracked positions at once, that's almost certainly a bad/partial
        # fetch rather than the user manually flattening everything. Skip and
        # let the next clean tick handle it.
        if ok and to_close and tm_symbols:
            share = len(to_close) / len(tm_symbols)
            if len(to_close) >= 5 and share >= 0.5:
                print(f"[TM] Reconcile: SKIPPED close of {len(to_close)}/"
                      f"{len(tm_symbols)} positions ({share:.0%}) — looks like "
                      f"a bad fetch, not real closes. Symbols: {sorted(to_close)}")
                ok = False  # suppress the close pass this tick
        if ok:
            for sym in to_close:
                # Use current price as the best-available exit price. Bybit
                # doesn't expose historical fill price for already-closed
                # positions via REST in a stable way.
                exit_price = self._get_current_price(sym)
                if exit_price is None:
                    # Fall back to the position's entry price so PnL=0 rather
                    # than a wild number. Better to be conservative.
                    with self._lock:
                        pos = self._positions.get(sym)
                    exit_price = pos['entry_price'] if pos else 0
                if exit_price > 0:
                    self._close_externally(sym, exit_price, reason='external_close')
                    closed_externally.append(sym)
        
        if adopted or closed_externally:
            print(f"[TM] Reconcile: adopted={adopted}, "
                  f"closed_externally={closed_externally}")
        return {
            'adopted': adopted,
            'closed_externally': closed_externally,
            'tm_count': len(tm_symbols),
            'live_count': len(live_symbols),
        }
    
    @staticmethod
    def _ff_score_snapshot(symbol: str) -> Optional[str]:
        """Fetch the Fuel Auto-Filter SCORE verdict string for `symbol` right
        now (e.g. 'STRONG HOLD 🟢▲ 79'), or None. Used to stamp a position at
        open and at close so the close recap can show the SCORE journey."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            return ff.score_snapshot(symbol) if ff else None
        except Exception:
            return None

    @staticmethod
    def _ff_exhaustion(symbol: str, side: str) -> Optional[float]:
        """Move-exhaustion % for `symbol`/`side` from the Fuel Auto-Filter
        (same value the panel shows). Used to stamp a position at open and at
        close so the close recap can show the Exhaust journey."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            return ff._exhaustion(symbol, side) if ff else None
        except Exception:
            return None

    @staticmethod
    def _ff_score_dict(symbol: str) -> Optional[Dict]:
        """Live Fuel Auto-Filter SCORE verdict dict ({score,label,color,dir,
        conflict}) for `symbol` — same shape the ⏱️ Active Timers rows carry, so
        the open-positions tables render the IDENTICAL badge."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            return ff.score_dict(symbol) if ff else None
        except Exception:
            return None

    @staticmethod
    def _ff_mm_snapshot(symbol: str) -> Optional[Dict]:
        """ММ (fuel) strength+direction snapshot for `symbol` right now:
        {'dir': 'LONG'|'SHORT'|None, 'str': int 0..100}. Stamped at open so the
        reason detail can show the ММ value at the moment the trade opened."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            if not ff:
                return None
            m = ff.get_fuel_strength_map().get(symbol.upper())
            if not m:
                return None
            return {'dir': m.get('dir'), 'str': m.get('now')}
        except Exception:
            return None

    @staticmethod
    def _ctr_snapshot(symbol: str) -> Optional[Dict]:
        """⚡ CTR state snapshot for `symbol` right now: {'stc','last_dir',
        'age_bars','zone'}. Stamped at OPEN and at CLOSE so the trade-history
        modal can show the CTR entry-gate reading at both moments."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return None
            c = (fe.get(symbol.upper()) or {}).get('ctr') or {}
            stc = c.get('stc')
            if stc is None:
                return None
            zone = 'OB' if stc >= 75 else ('OS' if stc <= 25 else 'MID')
            _age = c.get('last_signal_age_bars')
            return {'stc': stc, 'last_dir': c.get('last_dir'),
                    # both keys so the modal (last_signal_age_bars, like the chart
                    # badge / queue cell) and legacy readers (age_bars) both work.
                    'age_bars': _age, 'last_signal_age_bars': _age,
                    'tf': c.get('tf'), 'zone': zone}
        except Exception:
            return None

    def _decision_snapshot(self, symbol: str):
        """Decision-Center snapshot dict for `symbol` right now (headline,
        recommended, verdict…). Stamped at close (exit_decision) and computed
        live when viewing an open trade. Uses the SAME compute_decision as the
        ENTRY snapshot (entry_score) so вхід/вихід read identically — the old
        compute_bias(...)['decision'] path returned nothing (no such key), which
        left the «на виході» line blank. Heavy — call ONCE per close / per view,
        never in a loop."""
        try:
            price = self._get_current_price(symbol)
            if not price:
                return None
            return self.compute_decision(symbol, price)
        except Exception:
            return None

    @staticmethod
    def _mm_band_word(strength) -> str:
        """ММ market-maker-pressure band label (same bands as the UI): 0–10
        рівновага, 10–30 легкий, 30–60 помірний, 60–85 сильний, 85+ потужний
        тиск."""
        try:
            s = float(strength)
        except (TypeError, ValueError):
            return ''
        if s < 10:
            return 'рівновага'
        if s < 30:
            return 'легкий тиск'
        if s < 60:
            return 'помірний тиск'
        if s < 85:
            return 'сильний тиск'
        return 'потужний тиск'

    def _build_reason_detail(self, symbol: str, pos: Dict, reason: str,
                             pnl_pct: float, is_shadow: bool = False) -> str:
        """Compose a human-readable, information-rich close reason.

        Keeps the short `reason` code intact for stats/grouping; this is a
        separate descriptive string for the UI. Includes: a plain-language
        detail of WHY it closed, trade duration, peak/MFE (best unrealised
        PnL reached), and the SMC context (bias/zone + hold score) at close.
        """
        parts = []

        # 1. Plain-language detail of the trigger
        detail_map = {
            'stop_loss': 'Спрацював стоп-лосс',
            'take_profit': 'Досягнуто тейк-профіт',
            'trailing_stop': 'Трейлінг-стоп від піку',
            'bot_sl': '🤖 Бот-софт стоп-лосс (−%)',
            'bot_tp_trail': '🤖 Бот-софт трейлінг тейк-профіт (відкат від піку)',
            'time_stop': 'Закрито за часом (time-stop)',
            'htf_flip': 'HTF bias розвернувся проти позиції',
            'reverse_smc': 'SMC-структура розвернулась (CHoCH проти)',
            'ctr_reversal_after_peak': '🔄 Розворот після піку (CTR + підтвердження)',
            'reverse_signal': 'Протилежний сигнал входу',
            'forecast_1h_confluence': '1H прогноз проти позиції',
            'opposite_ob_exit': 'Ціна вдарилась у протилежний Order Block',
            'external_close': 'Закрито поза ботом (на біржі)',
            'auto_gate_wait': 'Закрито авто-гейтом по WAIT (немає вирівнювання ТФ)',
            'mm_below_min': 'ММ впало нижче порога закриття',
            'btc_flip': 'Банер ₿ BTCUSDT розвернувся проти позиції',
            'fuel_flipped': 'ММ розвернулось у протилежний бік',
            'fuel_faded': 'ММ згасло (втратило напрямок)',
            'potential_reached': 'Досягнуто виснаженість руху',
            'wait_verdict': 'Перехід у WAIT',
            'manual': 'Закрито вручну',
        }
        base = detail_map.get(reason)
        if base is None and reason.startswith('bos_') and reason.endswith('_partial'):
            n = reason.split('_')[1]
            base = f'Частковий вихід на BOS-{n}'
        parts.append(base or reason)

        # NB: duration / peak / ММ-at-open were removed from this text — that
        # recap now lives in the 📈 trade-history chart modal. Reason column
        # stays short: just WHY it closed.
        return ' · '.join(str(p) for p in parts if p)

    def _peak_pnl(self, symbol: str, pnl_pct: float, is_shadow: bool = False):
        """Best unrealised PnL% the trade reached (MFE), clamped to at least
        the final PnL so it never contradicts the result. Stored on close so
        the trade-history modal can show peak / give-back.

        Robust source: takes the MAX of the tracked peak, the recorded history
        samples (the SAME series the chart plots) and the final PnL. The tracked
        peak alone could drift to 0 (state reset on reconcile/restart) while the
        chart clearly showed a higher peak — deriving from samples keeps the
        header consistent with the plotted line."""
        state = (self._shadow_pos_state if is_shadow else self._pos_state).get(symbol, {})
        pos = (self._shadow_positions if is_shadow else self._positions).get(symbol) or {}
        cands = [pnl_pct]
        peak = state.get('peak_pnl_pct')
        if peak is not None:
            cands.append(peak)
        for h in (pos.get('history') or []):
            hp = h.get('pnl')
            if hp is not None:
                cands.append(hp)
        return round(max(cands), 4)

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return
        
        bybit_side = 'Buy' if pos['side'] == 'LONG' else 'Sell'
        qty = pos.get('remaining_qty', pos['qty'])
        
        try:
            self.bybit.close_position(symbol=symbol, side=bybit_side, qty=qty)
        except Exception as e:
            self._notify(f"⚠️ Close API error for {symbol}: {e}")
        
        # Calculate PnL
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        # Approximate USD PnL (qty is in coin units)
        pnl_usd = (exit_price - entry) * qty * (1 if pos['side'] == 'LONG' else -1)
        try:
            from detection.activity_log import log_activity
            _mh = self._fmt_manual_hist(pos)
            log_activity(symbol, 'closed',
                         f'{self._reason_label(reason)} · PnL {pnl_pct:+.2f}%'
                         + (f' · {_mh}' if _mh else ''),
                         side=pos['side'], source='TM',
                         extra={'manual_sl_hist': list(pos.get('manual_sl_hist') or []),
                                'manual_tp_hist': list(pos.get('manual_tp_hist') or [])})
        except Exception:
            pass

        closed = {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': entry,
            'exit_price': exit_price,
            'qty': pos['qty'],
            'remaining_qty_at_close': qty,
            'opened_at': pos['opened_at'],
            'closed_at': time.time(),
            'pnl_pct': round(pnl_pct, 4),
            'pnl_usd': round(pnl_usd, 2),
            'reason': reason,
            'reason_detail': self._build_reason_detail(
                symbol, pos, reason, round(pnl_pct, 4), is_shadow=False),
            'peak_pnl_pct': self._peak_pnl(symbol, round(pnl_pct, 4), is_shadow=False),
            'exit_decision': self._decision_snapshot(symbol),
            'ctr_open': pos.get('ctr_open'),
            'ctr_close': self._ctr_snapshot(symbol),
            # All Manual SL/TP levels that applied during the trade (+ final) —
            # shown in the trade-history modal.
            'manual_sl_hist': list(pos.get('manual_sl_hist') or []),
            'manual_tp_hist': list(pos.get('manual_tp_hist') or []),
            'manual_sl': pos.get('manual_sl'),
            'manual_tp': pos.get('manual_tp'),
            'opened_by': pos.get('opened_by', ''),
            'partial_closes_done': pos.get('partial_closes_done', []),
            # Carry the entry-side advisory snapshot into the closed record
            # so we can correlate "what the bot thought at open" with
            # "how the trade actually went". Critical for tuning weights
            # and validating whether Entry Score has predictive value.
            'entry_score': pos.get('entry_score'),
            # 🔬 Calibration fields (from the FF Q2 engine + monitors).
            'mae_pnl_pct': pos.get('mae_pnl_pct'),
            'ff_entry_score': pos.get('ff_entry_score'),
            'ff_queue_wait_sec': pos.get('ff_queue_wait_sec'),
            'ff_ctr_at_signal': pos.get('ff_ctr_at_signal'),
            'ff_ctr_at_open': pos.get('ff_ctr_at_open'),
            'ff_kind': pos.get('ff_kind'),
            # Mature Decision Center at open (for the entry-model comparison).
            'ff_dec_score': pos.get('ff_dec_score'),
            'ff_dec_reco': pos.get('ff_dec_reco'),
            'ff_dec_verdict': pos.get('ff_dec_verdict'),
            # Multi-TF CTR confluence at open (record-only, for validation).
            'ff_ctr_mtf_align': pos.get('ff_ctr_mtf_align'),
            'ff_ctr_mtf_trend': pos.get('ff_ctr_mtf_trend'),
            'ff_ctr_mtf_timing': pos.get('ff_ctr_mtf_timing'),
            # ₿ session at open — to validate opening during a ₿ pause.
            'ff_btc_at_open': pos.get('ff_btc_at_open'),
            'ff_btc_paused_at_open': pos.get('ff_btc_paused_at_open'),
        }

        with self._lock:
            self._positions.pop(symbol, None)
            self._pos_state.pop(symbol, None)
            closed['history'] = list(pos.get('history') or [])
            self._closed_trades.append(closed)
            if len(self._closed_trades) > CLOSED_TRADES_LIMIT:
                self._closed_trades = self._closed_trades[-CLOSED_TRADES_LIMIT:]
        self._persist_positions()
        self._persist_closed_trades()
        # The reduce-only close order already went out above — finalize in the
        # background (real-PnL fetch + archive + notify) so it NEVER delays the
        # close or stalls the monitor loop. The table shows the approximation
        # instantly, then updates to exchange-truth when the record settles.
        self._finalize_close_async(symbol, closed, pos['side'], pos,
                                   notify_fn=self._notify_close)

    def _partial_close(self, symbol: str, pct: float, exit_price: float, reason: str):
        """Close a percentage of remaining qty."""
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return
        
        remaining = pos.get('remaining_qty', pos['qty'])
        close_qty = remaining * (pct / 100)
        if close_qty <= 0:
            return
        
        bybit_side = 'Buy' if pos['side'] == 'LONG' else 'Sell'
        try:
            self.bybit.close_position(symbol=symbol, side=bybit_side, qty=close_qty)
        except Exception as e:
            print(f"[TM] Partial close error for {symbol}: {e}")
            return
        
        new_remaining = remaining - close_qty
        pos['remaining_qty'] = new_remaining
        self._persist_positions()
        
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        
        side_icon = '🟢' if pos['side'] == 'LONG' else '🔴'
        self._notify(
            f"{side_icon} <b>Partial close</b>: #{symbol}\n"
            f"📤 {pct:.0f}% of position closed\n"
            f"📍 Exit: {self._fmt_price(exit_price)}\n"
            f"💰 PnL: {pnl_pct:+.2f}%\n"
            f"🔖 Reason: {self._reason_label(reason)}\n"
            f"💼 Remaining: {new_remaining:.6g}"
        )
    
    def _partial_close_shadow(self, symbol: str, pct: float,
                              exit_price: float, reason: str):
        """Paper-trading mirror of _partial_close. No Bybit calls — purely
        book-keeping so shadow positions go through the same BOS-N partial
        lifecycle as real ones.
        
        Behaviour:
          - Reduces shadow_position.remaining_qty by `pct`%
          - Logs a `[TEST] Partial close` Telegram message
          - Does NOT create a separate closed_trade entry (that only happens
            when remaining_qty hits 0; mirrors real-side behavior).
          - When remaining_qty drops below 0.001 (effectively 0), we treat
            the position as fully closed via _close_shadow with the same reason.
        """
        with self._lock:
            pos = self._shadow_positions.get(symbol)
        if not pos:
            return
        
        remaining = pos.get('remaining_qty', pos.get('qty', 1.0))
        close_qty = remaining * (pct / 100)
        if close_qty <= 0:
            return
        
        new_remaining = remaining - close_qty
        # Floor very small remainders to fully close — avoids floating-point
        # crumbs leaving a 0.0001 ghost position open.
        if new_remaining < 0.001:
            # This partial actually closes the entire remaining position
            self._close_shadow(symbol, exit_price, reason=reason)
            return
        
        with self._lock:
            pos['remaining_qty'] = new_remaining
        self._persist_shadow_positions()
        
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        
        side_icon = '🟢' if pos['side'] == 'LONG' else '🔴'
        self._notify(
            f"{side_icon} <b>[TEST] Partial close</b>: #{symbol}\n"
            f"📤 {pct:.0f}% of paper position closed\n"
            f"📍 Exit: {self._fmt_price(exit_price)}\n"
            f"💰 PnL: {pnl_pct:+.2f}%\n"
            f"🔖 Reason: {self._reason_label(reason)}\n"
            f"💼 Remaining: {new_remaining * 100:.0f}%\n"
            f"🧪 Paper trade (no real close)",
            is_test=True,
        )
        print(f"[TM] [TEST] Shadow partial close: {symbol} {pos['side']} "
              f"{pct:.0f}% → remaining {new_remaining * 100:.0f}%")
    
    # ============================================================
    # Shadow (paper) positions — for test_mode
    # ============================================================
    
    def _open_shadow(self, symbol: str, side: str, entry_price: float, opened_by: str, bypass_gates: bool = False):
        """Open a paper-trading position. No Bybit calls.

        bypass_gates: If True, ignore LONG/SHORT entry gates (used by Fuel Auto-Filter).

        Returns a structured result like _open_position:
          {'ok': True}                    — shadow opened
          {'ok': False, 'reason': <str>}  — blocked, with why
        so on_signal can surface the real reason to the chart marker.
        """
        # === Global directional gate (same toggle as real) ===
        # Test mode shadows respect the same LONG/SHORT master gate so the
        # paper-trading view stays consistent with what a real deployment
        # would have done. Closures/exits run normally; only opens blocked.
        # EXCEPT: when bypass_gates=True (Fuel Auto-Filter), always allow.
        if not bypass_gates and not self._side_allowed(side):
            print(f"[TM] 🚫 SHADOW {side} entries disabled — {symbol} not opened")
            return {'ok': False, 'reason':
                    f'{side} entries are disabled (master {side} toggle OFF)'}

        # === Fuel Auto-Filter confirmation gate (same as real) ===
        # Paper trades must pass the same FF-table confirmation so the test view
        # mirrors what a real deployment would do.
        if not bypass_gates and self._settings.get('require_fuel_confirm', True):
            try:
                from detection.fuel_filter import get_fuel_filter
                ff = get_fuel_filter()
                if ff is None or not ff.is_in_table(symbol, side):
                    print(f"[TM] 🚫 Fuel-confirm gate: {symbol} {side} not in "
                          f"❤️ Fuel Auto-Filter table — SHADOW open ignored")
                    return {'ok': False, 'reason':
                            f'{symbol} {side} is not in the ❤️ Fuel Auto-Filter '
                            f'table yet (paper). Wait for it, or turn off '
                            f'"❤️ Підтвердження від Fuel Auto-Filter".'}
            except Exception as e:
                print(f"[TM] Fuel-confirm gate error (shadow) for {symbol}: {e} — open ignored")
                return {'ok': False, 'reason': f'Fuel-confirm gate error: {e}'}

        # Compute unified Decision Center verdict — same shape as real open
        decision = None
        try:
            decision = self.compute_decision(symbol, entry_price)
        except Exception as e:
            print(f"[TM] decision compute error on shadow open: {e}")
        opened_by_full = self._format_opened_by(opened_by, symbol,
                                                 entry_score=decision)
        # SL/TP levels — same calc as the real book — so the paper monitor's
        # hard SL/TP (now mirrored) actually has levels to check against.
        s = self._settings
        _sl = self._calc_sl_price(side, float(entry_price), symbol) if s.get('use_sl') else None
        _tp = self._calc_tp_price(side, float(entry_price)) if s.get('use_tp') else None
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': float(entry_price),
            'opened_at': time.time(),
            'opened_by': opened_by_full,
            'sl_price': float(_sl) if _sl else None,
            'tp_price': float(_tp) if _tp else None,
            'be_moved': False,
            'shadow': True,
            # === Fields needed for BOS-N partial closes on shadow positions ===
            # We track qty as a unit (1.0) and remaining_qty as a fraction,
            # so a "70% partial close" reduces remaining_qty from 1.0 to 0.3.
            # This is purely book-keeping — no Bybit calls — but it lets
            # shadow trades go through the same partial-close lifecycle as
            # real trades for accurate paper-trading data.
            'qty': 1.0,
            'remaining_qty': 1.0,
            'partial_closes_done': [],       # ['bos_2', 'bos_3', ...]
            'bos_count_since_entry': 1,      # opening BOS counts as #1 (matches real)
            'trailing_active': False,
            'trailing_peak': float(entry_price),
            'trailing_via_bos2': False,
        }
        if decision is not None:
            pos['entry_score'] = decision
        # FF SCORE + Exhaustion at open (kept for analytics); ММ at open for Reason.
        pos['ff_score_open'] = self._ff_score_snapshot(symbol)
        pos['ff_exh_open'] = self._ff_exhaustion(symbol, side)
        pos['ff_mm_open'] = self._ff_mm_snapshot(symbol)
        pos['ctr_open'] = self._ctr_snapshot(symbol)
        with self._lock:
            self._shadow_positions[symbol] = pos
            self._shadow_pos_state[symbol] = self._fresh_pos_state()
        self._persist_shadow_positions()
        
        # Use colored circle for direction (instead of 📊)
        icon = '🟢' if side == 'LONG' else '🔴'
        # Single-line decision summary instead of multi-line breakdown
        es_block = self._format_entry_score_telegram(decision)
        # Order Block info — informational for now. Computed on demand from
        # the scanner's klines cache. Failure to compute is non-fatal — we
        # just skip the OB line in the message.
        ob_line = self._format_last_ob_telegram(symbol)
        # Source WITHOUT the decision suffix (the 🤪 verdict is shown once,
        # below in es_block — fixes the duplicated «LONG 83% (marginal)»).
        src = self._base_opened_by(opened_by_full)
        msg = (
            f"{icon} ▶️ <b>ВІДКРИТО {side}</b> · #{symbol}   🧪 ТЕСТ\n"
            f"━━━━━━━━━━━━\n"
            f"📍 Вхід: {self._fmt_price(entry_price)}\n"
            f"{es_block}"
            f"{ob_line}"
            f"📋 Джерело: {src}"
        )
        self._notify(msg, is_test=True, category='trades')
        print(f"[TM] [TEST] Shadow open: {symbol} {side} @ {self._fmt_price(entry_price)}")
        return {'ok': True}

    def _format_last_ob_telegram(self, symbol: str) -> str:
        """Render a one-line Order Block status for Telegram OPEN messages.
        
        Reads from DB cache (same source the OB Filter signal gate uses),
        falling back to inline scanner-cache compute only when DB row is
        absent. Format:
        
            🎯 OB Confirm: 🟢 LONG (1H) — when filter is on and OB matched
            🎯 OB: 🟢 LONG (1H) zone 103,400—103,520 — info-only mode
        
        Returns empty string when no OB exists. Non-fatal — designed so
        callers can splice the result into a message without conditionals.
        
        For OPEN messages this implicitly tells the user "the gate let
        this through" because if filter were on and OB didn't match, the
        signal would never have reached _open_*() in the first place.
        """
        if not self.scanner:
            return ''
        
        # Read OB from DB — same row that gated the signal
        try:
            from storage.db_operations import get_db
            ob_tf = self.scanner._settings.get('ob_filter_timeframe', '1h')
            filter_on = bool(self.scanner._settings.get('ob_filter_enabled', False))
            row = get_db().get_smc_ob_state(symbol, ob_tf)
        except Exception as e:
            print(f"[TM] OB DB lookup error for {symbol}: {e}")
            return ''
        
        # If DB row is missing, try the inline path (scanner just started)
        if row is None or not row.get('bias'):
            try:
                from detection.ob_detector import detect_last_order_block
                with self.scanner._lock:
                    cached = dict(self.scanner._cache.get(symbol) or {})
                klines = cached.get('klines') or []
                analysis = cached.get('analysis') or {}
                if klines:
                    internal = analysis.get('internal') or {}
                    klines_closed = klines[:-1] if len(klines) >= 2 else klines
                    ob = detect_last_order_block(
                        klines=klines_closed,
                        pivots=internal.get('pivots', []),
                        events=internal.get('events', []),
                    )
                    if ob:
                        row = {
                            'bias': ob['bias'],
                            'bar_high': ob['bar_high'],
                            'bar_low': ob['bar_low'],
                            'bar_time': ob['bar_time'],
                        }
            except Exception:
                pass
        
        if not row or not row.get('bias'):
            return ''
        
        bias = row['bias']
        bias_label = 'LONG' if bias == 'BULLISH' else 'SHORT'
        bias_icon = '🟢' if bias == 'BULLISH' else '🔴'
        tf_label = ob_tf.upper()
        
        # Age — bar_time is ms epoch
        bar_time_ms = row.get('bar_time') or 0
        age_str = ''
        if bar_time_ms:
            try:
                age_secs = max(0, time.time() - (bar_time_ms / 1000))
                if age_secs < 3600:
                    age_str = f" (~{int(age_secs / 60)}m ago)"
                elif age_secs < 86400:
                    age_str = f" (~{int(age_secs / 3600)}h ago)"
                else:
                    age_str = f" (~{int(age_secs / 86400)}d ago)"
            except Exception:
                pass
        
        # Indicate freshness (CHoCH-created = ★ fresh trend reversal,
        # the only case the strict gate allows when filter is on).
        created_by = (row.get('created_by_tag') or '').upper()
        if created_by == 'CHOCH':
            fresh_mark = ' ★ fresh CHoCH'
        elif created_by == 'BOS':
            fresh_mark = ' (BOS continuation)'
        else:
            fresh_mark = ''
        
        if filter_on:
            # Filter mode: short confirmation line — the gate already
            # validated direction match AND CHoCH creation, so we just
            # affirm both for the user. Filter wouldn't have let this
            # signal through if the OB wasn't fresh.
            return (f"🎯 OB Confirm: {bias_icon} {bias_label} "
                    f"({tf_label}){fresh_mark}{age_str}\n")
        else:
            # Info-only mode: also show the zone bounds for context
            zone = (f"{self._fmt_price(row['bar_low'])}—"
                    f"{self._fmt_price(row['bar_high'])}")
            return (f"🎯 OB: {bias_icon} {bias_label} ({tf_label}) "
                    f"zone {zone}{fresh_mark}{age_str}\n")
    
    def _format_entry_score_telegram(self, entry_score: Optional[Dict]) -> str:
        """Single-line Decision Center summary for Telegram OPEN messages.
        
        Was previously a multi-line breakdown — replaced with a compact
        one-liner per user feedback that the verbose form was confusing.
        Produces output like:
        
            🤪 Decision: LONG 78% (good)
            
        Empty string when entry_score is missing/disabled. Despite the
        legacy method name, this now reads from the Decision Center
        verdict shape (which extends entry_score with headline/verdict).
        """
        if not entry_score:
            return ''
        headline = entry_score.get('headline')
        verdict = self._verdict_ua(entry_score.get('verdict', '?'))
        if headline:
            return f"🤪 Рішення: {headline} ({verdict})\n"
        score = entry_score.get('score', 0)
        sign = '+' if score >= 0 else ''
        return f"🤪 Рішення: {sign}{score:.0f} ({verdict})\n"
    
    def _format_entry_score_recap(self, entry_score: Optional[Dict]) -> str:
        """Compact one-line recap of the Decision Center verdict for
        Telegram CLOSE messages. Format:
        
            🤪 Was: LONG 78% (good)
        """
        if not entry_score:
            return ''
        headline = entry_score.get('headline')
        verdict = self._verdict_ua(entry_score.get('verdict', '?'))
        if headline:
            return f"🤪 Було: {headline} ({verdict})\n"
        score = entry_score.get('score', 0)
        sign = '+' if score >= 0 else ''
        return f"🤪 Було: {sign}{score:.0f} ({verdict})\n"

    @staticmethod
    def _verdict_ua(v):
        return {'good': 'сильний', 'marginal': 'середній',
                'poor': 'слабкий'}.get(str(v or '').lower(), v or '')
    
    def _close_shadow(self, symbol: str, exit_price: float, reason: str):
        """Close a paper position. No Bybit calls — Telegram only."""
        with self._lock:
            pos = self._shadow_positions.get(symbol)
        if not pos:
            return
        
        entry = pos['entry_price']
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        try:
            from detection.activity_log import log_activity
            _mh = self._fmt_manual_hist(pos)
            log_activity(symbol, 'closed',
                         f'{self._reason_label(reason)} · PnL {pnl_pct:+.2f}% (paper)'
                         + (f' · {_mh}' if _mh else ''),
                         side=pos['side'], source='TM',
                         extra={'manual_sl_hist': list(pos.get('manual_sl_hist') or []),
                                'manual_tp_hist': list(pos.get('manual_tp_hist') or [])})
        except Exception:
            pass

        closed = {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': entry,
            'exit_price': float(exit_price),
            'opened_at': pos['opened_at'],
            'closed_at': time.time(),
            'pnl_pct': round(pnl_pct, 4),
            'reason': reason,
            'reason_detail': self._build_reason_detail(
                symbol, pos, reason, round(pnl_pct, 4), is_shadow=True),
            'peak_pnl_pct': self._peak_pnl(symbol, round(pnl_pct, 4), is_shadow=True),
            'exit_decision': self._decision_snapshot(symbol),
            'ctr_open': pos.get('ctr_open'),
            'ctr_close': self._ctr_snapshot(symbol),
            # All Manual SL/TP levels that applied during the trade (+ final) —
            # shown in the trade-history modal.
            'manual_sl_hist': list(pos.get('manual_sl_hist') or []),
            'manual_tp_hist': list(pos.get('manual_tp_hist') or []),
            'manual_sl': pos.get('manual_sl'),
            'manual_tp': pos.get('manual_tp'),
            'opened_by': pos.get('opened_by', ''),
            'shadow': True,
            # Same as real-position close — preserve the entry snapshot for
            # post-mortem analysis (was the entry score predictive?)
            'entry_score': pos.get('entry_score'),
            # 🔬 Calibration fields (from the FF Q2 engine + monitors).
            'mae_pnl_pct': pos.get('mae_pnl_pct'),
            'ff_entry_score': pos.get('ff_entry_score'),
            'ff_queue_wait_sec': pos.get('ff_queue_wait_sec'),
            'ff_ctr_at_signal': pos.get('ff_ctr_at_signal'),
            'ff_ctr_at_open': pos.get('ff_ctr_at_open'),
            'ff_kind': pos.get('ff_kind'),
            # Mature Decision Center at open (for the entry-model comparison).
            'ff_dec_score': pos.get('ff_dec_score'),
            'ff_dec_reco': pos.get('ff_dec_reco'),
            'ff_dec_verdict': pos.get('ff_dec_verdict'),
            # Multi-TF CTR confluence at open (record-only, for validation).
            'ff_ctr_mtf_align': pos.get('ff_ctr_mtf_align'),
            'ff_ctr_mtf_trend': pos.get('ff_ctr_mtf_trend'),
            'ff_ctr_mtf_timing': pos.get('ff_ctr_mtf_timing'),
            # ₿ session at open — to validate opening during a ₿ pause.
            'ff_btc_at_open': pos.get('ff_btc_at_open'),
            'ff_btc_paused_at_open': pos.get('ff_btc_paused_at_open'),
        }

        with self._lock:
            self._shadow_positions.pop(symbol, None)
            self._shadow_pos_state.pop(symbol, None)
            closed['history'] = list(pos.get('history') or [])
            self._shadow_closed.append(closed)
            if len(self._shadow_closed) > CLOSED_TRADES_LIMIT:
                self._shadow_closed = self._shadow_closed[-CLOSED_TRADES_LIMIT:]
        self._persist_shadow_positions()
        self._persist_shadow_closed()
        self._archive_closed(closed, pos, is_paper=True)
        
        is_win = pnl_pct > 0
        icon = '✅' if is_win else '❌'
        # Recap the entry score so the user sees how the bot's pre-trade
        # assessment correlates with the actual outcome. This is what makes
        # post-mortem analysis useful — "I closed at -2%, but Entry Score was
        # 'good +50' — should I trust this score or down-weight it?"
        es_recap = self._format_entry_score_recap(pos.get('entry_score'))
        peak_line = self._fmt_peak_line(closed)
        msg = (
            f"{icon} ⏹ <b>ЗАКРИТО {pos['side']}</b> {pnl_pct:+.2f}% · #{symbol}   🧪 ТЕСТ\n"
            f"━━━━━━━━━━━━\n"
            f"📍 Вхід → Вихід: {self._fmt_price(entry)} → {self._fmt_price(exit_price)}\n"
            f"🔖 Причина: {self._reason_label(reason)}\n"
            f"{peak_line}"
            f"{es_recap}"
        ).rstrip()
        self._notify(msg, is_test=True, category='trades')
        print(f"[TM] [TEST] Shadow close: {symbol} {pos['side']} @ {self._fmt_price(exit_price)} "
              f"({pnl_pct:+.2f}% reason={reason})")
    
    # ============================================================
    # Position sizing
    # ============================================================
    
    def _calculate_qty(self, symbol: str, entry_price: float) -> float:
        s = self._settings
        mode = s.get('sizing_mode', 'fixed_usd')
        
        if mode == 'fixed_usd':
            usd = float(s.get('fixed_usd_amount', 100))
            qty = usd / entry_price
        elif mode == 'fixed_pct':
            balance = self._get_balance()
            pct = float(s.get('fixed_pct_balance', 2.0))
            usd = balance * (pct / 100)
            qty = usd / entry_price
        elif mode == 'risk_based':
            balance = self._get_balance()
            risk_pct = float(s.get('risk_pct_balance', 1.0)) / 100
            sl_pct = float(s.get('sl_pct', 2.0)) / 100
            if sl_pct <= 0:
                return 0
            # Risk USD = balance * risk_pct
            # Loss per coin at SL = entry * sl_pct
            # qty = risk_usd / loss_per_coin
            risk_usd = balance * risk_pct
            qty = risk_usd / (entry_price * sl_pct)
        else:
            qty = 0
        
        if qty <= 0:
            return 0
        
        # === Round qty to the instrument's lot size ===
        # Bybit rejects orders whose qty doesn't match the symbol's qtyStep
        # with "Qty invalid" (ErrCode 10001). E.g. ENAUSDT requires integer
        # qty, so 188.84 fails — must be 188. We round DOWN to the nearest
        # step (never risk over-sizing) and enforce minOrderQty.
        qty = self._round_qty_to_lot(symbol, qty)
        return max(0, qty)
    
    def _round_qty_to_lot(self, symbol: str, qty: float) -> float:
        """Round an order quantity DOWN to the symbol's qtyStep and enforce
        minOrderQty. Falls back to coarse decimal rounding if instrument
        info is unavailable (keeps old behavior as a safety net).
        """
        import math
        lot = None
        try:
            if self.bybit and hasattr(self.bybit, 'get_lot_size'):
                lot = self.bybit.get_lot_size(symbol)
        except Exception as e:
            print(f"[TM] get_lot_size warn for {symbol}: {e}")
        
        if lot and lot.get('qty_step', 0) > 0:
            step = lot['qty_step']
            min_qty = lot.get('min_qty', 0)
            # Floor to step. Add tiny epsilon before floor to avoid float
            # artifacts (e.g. 188.9999999 from 1889 * 0.1).
            steps = math.floor((qty + step * 1e-9) / step)
            rounded = steps * step
            # Determine decimal places from the step so we don't reintroduce
            # float noise (e.g. 0.1 * 1888 = 188.8000000002).
            step_str = ('%.10f' % step).rstrip('0').rstrip('.')
            decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
            rounded = round(rounded, decimals)
            # Enforce minimum order qty — if below min, the trade can't open
            # at this size. Return 0 so caller treats it as a sizing failure.
            if min_qty > 0 and rounded < min_qty:
                print(f"[TM] {symbol} qty {rounded} below minOrderQty "
                      f"{min_qty} — cannot open at this size")
                return 0
            return rounded
        
        # Fallback (no instrument info) — old coarse rounding
        if qty < 1:
            return round(qty, 6)
        return round(qty, 3)
    
    
    def _calc_sl_price(self, side: str, entry: float,
                         symbol: Optional[str] = None) -> float:
        """Compute SL price using the configured strategy. Falls back to
        percentage mode on any failure so the trade still opens with a
        sane stop instead of being blocked or sent without protection."""
        sl_mode = self._settings.get('sl_mode', 'pct')
        
        if sl_mode == 'volumized_ob' and symbol:
            sl = self._calc_sl_from_volumized_ob(side, entry, symbol)
            if sl is not None:
                return sl
            # Fall through with logging done inside the helper
        
        # Default: percentage offset from entry
        sl_pct = self._settings.get('sl_pct', 2.0) / 100
        if side == 'LONG':
            return entry * (1 - sl_pct)
        return entry * (1 + sl_pct)
    
    def _read_volumized_meta_from_cache(self, symbol: str) -> Optional[Dict]:
        """Pull pre-computed Volumized OB meta from the scanner's cache.
        Returns the meta dict (with ob_type/top/bottom) or None when:
          - Scanner has no entry for this symbol
          - Scanner's Volumized filter has been off long enough that the
            cache is stale or missing
          - Meta lacks required keys (corrupt or pre-init state)
        Cheap (just a dict lookup); used as the fast path before falling
        through to on-demand recompute."""
        if not self.scanner:
            return None
        try:
            vol_cache = self.scanner._volumized_trend_cache.get(symbol, {})
        except Exception:
            return None
        meta = vol_cache.get('meta') or {}
        # Treat half-populated meta as "no meta" so the on-demand path
        # gets a chance to compute fresh data.
        if not all(k in meta for k in ('ob_type', 'top', 'bottom')):
            return None
        return meta
    
    def _compute_volumized_meta_on_demand(self, symbol: str) -> Optional[Dict]:
        """Compute Volumized OB meta from klines RIGHT NOW, independent of
        the scanner's Vol-OB filter setting. This makes the SL Vol-OB
        feature usable even when the user has the Volumized OB Trend
        filter turned off in SMC settings.
        
        Steps:
          1. Pull klines on volumized_timeframe (default 1h) from scanner's
             cache for this symbol (no extra API call when the scanner has
             already fetched them).
          2. If scanner doesn't have klines for this TF, return None — we
             refuse to do a fresh HTTP fetch here because this runs on the
             trade-open hot path and an extra network round-trip would
             delay order placement. The pct fallback is acceptable.
          3. Run `get_latest_ob_trend()` with the same parameters the
             scanner would use, so the OB matches what's displayed on the
             chart's Vol OB badge.
        
        Returns the trend_meta dict (with ob_type/top/bottom) or None on
        any failure."""
        if not self.scanner:
            return None
        try:
            scn_settings = self.scanner._settings or {}
        except Exception:
            scn_settings = {}
        vol_tf = scn_settings.get('volumized_timeframe', '1h')
        
        # Grab klines from the scanner's per-TF cache. The scanner's main
        # loop populates this whenever it scans the symbol on any TF, so
        # we usually have something to work with.
        try:
            tf_cache = self.scanner._tf_klines_cache.get(symbol, {}) \
                if hasattr(self.scanner, '_tf_klines_cache') else {}
        except Exception:
            tf_cache = {}
        klines = (tf_cache.get(vol_tf, {}) or {}).get('klines')
        
        # Secondary source: if main TF happens to match vol_tf, the symbol
        # cache might have klines directly (scanner._cache[symbol]).
        if not klines:
            try:
                main_cache = self.scanner._cache.get(symbol, {}) \
                    if hasattr(self.scanner, '_cache') else {}
                main_tf = scn_settings.get('timeframe', '15m')
                if main_tf == vol_tf:
                    klines = main_cache.get('klines')
            except Exception:
                pass
        
        if not klines or len(klines) < 60:
            return None  # Not enough history — abandon to pct fallback
        
        try:
            from detection.volumized_ob import get_latest_ob_trend
            vol_result = get_latest_ob_trend(
                klines,
                swing_length=int(scn_settings.get('volumized_swing_length', 10)),
                ob_end_method=scn_settings.get('volumized_ob_end_method', 'Wick'),
                max_atr_mult=float(scn_settings.get('volumized_max_atr_mult', 3.5)),
                zone_count=scn_settings.get('volumized_zone_count', 'Low'),
                combine_obs=bool(scn_settings.get('volumized_combine_obs', True)),
            )
        except Exception as e:
            print(f"[TM] {symbol}: on-demand Vol-OB compute error: {e}")
            return None
        
        meta = vol_result.get('trend_meta') or {}
        if not all(k in meta for k in ('ob_type', 'top', 'bottom')):
            return None
        print(f"[TM] {symbol}: Vol-OB meta computed on-demand "
              f"({meta.get('ob_type')} {meta.get('bottom'):.6f}–{meta.get('top'):.6f})")
        return meta
    
    def _calc_sl_from_volumized_ob(self, side: str, entry: float,
                                     symbol: str) -> Optional[float]:
        """Compute SL from the Volumized OB block boundary.
        
            LONG  → ob_bottom × (1 − buffer_pct/100)
            SHORT → ob_top    × (1 + buffer_pct/100)
        
        Source of the OB: prefer the SMC scanner's `_volumized_trend_cache`
        (fast path — already computed during the scan tick). When that cache
        isn't populated — e.g., the scanner's Volumized OB Trend filter is
        OFF, or this symbol hasn't been scanned yet — we compute the OB
        on-the-fly here using the scanner's klines cache and the same
        `get_latest_ob_trend()` function. This makes the SL mode a
        standalone, self-sufficient feature: works regardless of the
        scanner's filter setting, just like the user expects.
        
        Returns None — and the caller falls back to pct mode — if:
          - Scanner reference unavailable (TM started without scanner)
          - On-demand compute also fails (no klines available, ATR warmup
            not done, no swings detected, etc.)
          - Latest OB direction doesn't match the trade direction (LONG
            signal but latest OB is Bear — rare race where Vol direction
            flipped between signal fire and position open)
          - Computed SL would be on the WRONG side of entry (would close
            instantly on market open — sanity guard)
        """
        if not self.scanner:
            print(f"[TM] {symbol}: SL=Vol-OB but scanner ref unavailable "
                  f"— falling back to pct")
            return None
        
        # ---- Fast path: read pre-computed cache (filter was on during scan) ----
        meta = self._read_volumized_meta_from_cache(symbol)
        
        # ---- Slow path: cache empty → compute on-demand from klines ----
        if not meta:
            meta = self._compute_volumized_meta_on_demand(symbol)
        
        if not meta:
            print(f"[TM] {symbol}: SL=Vol-OB no OB meta available "
                  f"(cache + on-demand both failed) — falling back to pct")
            return None
        
        ob_type = meta.get('ob_type')           # 'Bull' / 'Bear'
        ob_top = meta.get('top')
        ob_bottom = meta.get('bottom')
        
        if ob_top is None or ob_bottom is None or ob_type is None:
            print(f"[TM] {symbol}: SL=Vol-OB meta incomplete — falling back to pct")
            return None
        
        # Direction sanity: LONG trades anchor on Bull OB, SHORT on Bear OB.
        # Mismatch typically means the Volumized trend flipped between
        # signal fire and position open.
        expected_type = 'Bull' if side == 'LONG' else 'Bear'
        if ob_type != expected_type:
            print(f"[TM] {symbol}: SL=Vol-OB but latest OB is {ob_type} "
                  f"and trade is {side} — direction mismatch, falling "
                  f"back to pct")
            return None
        
        buffer_pct = self._settings.get('sl_vob_buffer_pct', 0.2) / 100
        
        if side == 'LONG':
            sl = ob_bottom * (1 - buffer_pct)
            # SL must be BELOW entry for LONG. If the OB bottom (with
            # buffer) sits at or above entry, it can't function as a
            # stop — would close instantly. Fall back to pct.
            if sl >= entry:
                print(f"[TM] {symbol} LONG: SL=Vol-OB ({sl:.6f}) ≥ entry "
                      f"({entry:.6f}); OB bottom above entry, fallback to pct")
                return None
        else:  # SHORT
            sl = ob_top * (1 + buffer_pct)
            if sl <= entry:
                print(f"[TM] {symbol} SHORT: SL=Vol-OB ({sl:.6f}) ≤ entry "
                      f"({entry:.6f}); OB top below entry, fallback to pct")
                return None
        
        print(f"[TM] {symbol} {side}: SL=Vol-OB → {sl:.6f} "
              f"(OB {ob_type} {ob_bottom:.6f}–{ob_top:.6f}, "
              f"buffer {buffer_pct*100:.2f}%)")
        return sl
    
    def _calc_tp_price(self, side: str, entry: float) -> float:
        tp_pct = self._settings.get('tp_pct', 5.0) / 100
        if side == 'LONG':
            return entry * (1 + tp_pct)
        return entry * (1 - tp_pct)
    
    # ============================================================
    # Bybit helpers
    # ============================================================
    
    def _get_balance(self) -> float:
        try:
            return self.bybit.get_wallet_balance() or 0
        except Exception as e:
            print(f"[TM] balance fetch error: {e}")
            return 0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        # Prefer scanner's cached klines (avoids Bybit API hit)
        if self.scanner:
            try:
                p = self.scanner._get_live_price(symbol)
                if p:
                    return p
            except:
                pass
        # Fallback to Bybit
        try:
            p = self.bybit.get_price(symbol)
            if p > 0:
                return p
        except:
            pass
        return None
    
    def _update_exchange_sl(self, symbol: str, sl_price: float):
        try:
            self.bybit.set_trading_stop(symbol=symbol, stop_loss=sl_price)
        except Exception as e:
            print(f"[TM] update SL error for {symbol}: {e}")
    
    # ============================================================
    # Tradeable list (mirrored from scanner watchlist)
    # ============================================================
    
    def _is_tradeable(self, symbol: str) -> bool:
        """Check if symbol is in scanner's tradeable list."""
        if not self.scanner:
            return False
        try:
            tradeable = self.scanner.get_tradeable_symbols()
            return symbol in tradeable
        except:
            return False

    def _at_max_positions(self) -> bool:
        """Check if we've reached the max_open_positions limit for REAL trades.
        Returns True if limit reached, False otherwise."""
        max_pos = self._settings.get('max_open_positions', 10)
        with self._lock:
            current_count = len(self._positions)
        return current_count >= max_pos

    # ============================================================
    # State / queries
    # ============================================================
    
    def _compute_hold_score(self, pos: Dict) -> Optional[Dict]:
        """SMC Hold-Confidence for one open position. Cached per-symbol for
        HOLD_SCORE_TTL seconds. Returns the analyzer dict or None on failure
        (UI then simply shows no badge). Pure analysis — never acts."""
        HOLD_SCORE_TTL = 45
        sym = pos.get('symbol')
        if not sym:
            return None
        now = time.time()
        cached = self._hold_cache.get(sym)
        if cached and (now - cached[0]) < HOLD_SCORE_TTL:
            return cached[1]
        try:
            from detection.hold_analyzer import analyze_hold
            # LTF klines from the scanner cache (the TF the bot trades).
            ltf = None
            htf = None
            if self.scanner is not None:
                getc = getattr(self.scanner, '_get_cached_klines', None)
                if callable(getc):
                    ltf = getc(sym)
                else:
                    # fall back to raw cache structure used elsewhere
                    cache = getattr(self.scanner, '_cache', {})
                    entry = cache.get(sym) or {}
                    ltf = entry.get('klines')
                gethtf = getattr(self.scanner, '_get_cached_htf_klines', None)
                if callable(gethtf):
                    htf = gethtf(sym)
            if not ltf:
                return None
            result = analyze_hold(pos, ltf, htf)
            if not result.get('ok'):
                self._hold_cache[sym] = (now, None)
                return None
            self._hold_cache[sym] = (now, result)
            return result
        except Exception as e:
            print(f"[TM] hold-score error for {sym}: {e}")
            return None

    def _refresh_live_positions(self, force: bool = False):
        """Pull a fresh exchange-truth snapshot of open positions from Bybit,
        throttled so frequent UI polls don't hammer the API. The reconcile loop
        also refreshes this cache on its own cadence; this keeps it current
        between reconcile ticks so the UI never shows stale exchange numbers.
        """
        if not self.bybit or not self._positions:
            return
        now = time.time()
        if not force and (now - self._live_positions_at) < self._live_refresh_min_interval:
            return
        try:
            if hasattr(self.bybit, 'get_positions_checked'):
                live, ok = self.bybit.get_positions_checked()
            else:
                live, ok = (self.bybit.get_positions() or []), True
            if ok:
                self._live_positions = {p['symbol']: p for p in live
                                        if p.get('symbol')}
                self._live_positions_at = now
        except Exception as e:
            print(f"[TM] live positions refresh error: {e}")

    def get_state(self) -> Dict:
        # Refresh the exchange-truth snapshot (throttled) BEFORE taking the lock
        # so open-position figures match Bybit exactly, not our approximation.
        self._refresh_live_positions()
        with self._lock:
            positions = []
            for sym, pos in self._positions.items():
                live = self._live_positions.get(sym)
                # Exchange truth wins: entry = Bybit avgPrice, qty = Bybit size,
                # current = Bybit markPrice, PnL = Bybit unrealisedPnl. We only
                # fall back to our own values when the live snapshot is missing
                # (e.g. API down) — never invent numbers.
                if live and live.get('entry_price', 0) > 0:
                    entry = live['entry_price']
                    live_qty = live.get('size') or pos.get('qty')
                    mark = live.get('mark_price') or 0
                    current = mark if mark > 0 else (
                        self._get_current_price(sym) or entry)
                    unreal = live.get('unrealized_pnl')
                else:
                    entry = pos['entry_price']
                    live_qty = pos.get('qty')
                    current = self._get_current_price(sym) or entry
                    unreal = None
                if pos['side'] == 'LONG':
                    pnl_pct = (current - entry) / entry * 100
                else:
                    pnl_pct = (entry - current) / entry * 100
                synced = bool(live and live.get('entry_price', 0) > 0)
                pos_dict = {
                    **pos,
                    'entry_price': entry,
                    'qty': live_qty,
                    'current_price': current,
                    'pnl_pct': round(pnl_pct, 3),
                    # Source flag the UI reads to show a ⇄ "synced with Bybit"
                    # marker. 'bybit' = exchange-truth, 'approx' = our fallback.
                    'pnl_source': 'bybit' if synced else 'approx',
                }
                if unreal is not None:
                    pos_dict['pnl_usd'] = round(unreal, 4)
                if synced:
                    # Exchange size is the truth for the displayed qty too —
                    # the UI prefers remaining_qty, so keep it in sync.
                    pos_dict['remaining_qty'] = live_qty
                if live:
                    pos_dict['mark_price'] = live.get('mark_price')
                    pos_dict['liq_price'] = live.get('liq_price')
                    pos_dict['leverage'] = live.get('leverage')
                    pos_dict['position_value'] = live.get('position_value')
                # Attach Health Score (None when feature disabled)
                health = self._compute_health(pos, is_shadow=False)
                if health is not None:
                    pos_dict['health'] = health
                # Attach SMC Hold-Confidence (None on failure → no badge)
                hold = self._compute_hold_score(pos_dict)
                if hold is not None:
                    pos_dict['hold'] = hold
                positions.append(_lite_trade(pos_dict))
            
            closed = [_lite_trade(c) for c in self._closed_trades[-50:]]

            # Stats — real
            total_closed = len(self._closed_trades)
            wins = sum(1 for c in self._closed_trades if c.get('pnl_pct', 0) > 0)
            losses = sum(1 for c in self._closed_trades if c.get('pnl_pct', 0) < 0)
            total_pnl = sum(c.get('pnl_usd', 0) for c in self._closed_trades)
            
            # Shadow positions snapshot
            shadow_positions = []
            for sym, pos in self._shadow_positions.items():
                current = self._get_current_price(sym) or pos['entry_price']
                entry = pos['entry_price']
                if pos['side'] == 'LONG':
                    pnl_pct = (current - entry) / entry * 100
                else:
                    pnl_pct = (entry - current) / entry * 100
                pos_dict = {
                    **pos,
                    'current_price': current,
                    'pnl_pct': round(pnl_pct, 3),
                }
                health = self._compute_health(pos, is_shadow=True)
                if health is not None:
                    pos_dict['health'] = health
                hold = self._compute_hold_score(pos_dict)
                if hold is not None:
                    pos_dict['hold'] = hold
                shadow_positions.append(_lite_trade(pos_dict))
            shadow_closed = [_lite_trade(c) for c in self._shadow_closed[-50:]]
            
            # Stats — shadow
            sh_total = len(self._shadow_closed)
            sh_wins = sum(1 for c in self._shadow_closed if c.get('pnl_pct', 0) > 0)
            sh_losses = sum(1 for c in self._shadow_closed if c.get('pnl_pct', 0) < 0)
            sh_avg_pnl = (sum(c.get('pnl_pct', 0) for c in self._shadow_closed) / sh_total) if sh_total else 0
            
            # Entry Score validation stats — break down win rate by verdict
            # so the user can see whether the scorer's "good" predictions
            # actually outperform "marginal" / "poor" picks. This is the
            # core feedback loop for tuning weights and threshold.
            entry_stats = self._compute_entry_score_stats(self._closed_trades)
            entry_stats_shadow = self._compute_entry_score_stats(self._shadow_closed)
            
            return {
                'enabled': self.is_enabled(),
                'running': self._running,
                'tick_count': self._tick_count,
                'errors': self._errors,
                'positions': positions,
                'closed_trades': closed,
                'stats': {
                    'total_closed': total_closed,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': round(wins / total_closed * 100, 1) if total_closed else 0,
                    'total_pnl_usd': round(total_pnl, 2),
                    'entry_score_breakdown': entry_stats,
                },
                'shadow_positions': shadow_positions,
                'shadow_closed': shadow_closed,
                'shadow_stats': {
                    'total_closed': sh_total,
                    'wins': sh_wins,
                    'losses': sh_losses,
                    'win_rate': round(sh_wins / sh_total * 100, 1) if sh_total else 0,
                    'avg_pnl_pct': round(sh_avg_pnl, 2),
                    'entry_score_breakdown': entry_stats_shadow,
                },
                'settings': dict(self._settings),
            }
    
    @staticmethod
    def _compute_entry_score_stats(closed_trades: List[Dict]) -> Dict:
        """For each verdict bucket (good / marginal / poor / unknown),
        compute count, win rate, average PnL%. The 'unknown' bucket
        catches trades that closed before Entry Score existed (or while
        the feature was disabled), so older history doesn't skew totals.
        """
        buckets = {'good': [], 'marginal': [], 'poor': [], 'unknown': []}
        for c in closed_trades:
            es = c.get('entry_score') or {}
            v = es.get('verdict') if isinstance(es, dict) else None
            bucket = v if v in ('good', 'marginal', 'poor') else 'unknown'
            buckets[bucket].append(c)
        
        result = {}
        for name, trades in buckets.items():
            n = len(trades)
            if n == 0:
                result[name] = {'count': 0, 'win_rate': None, 'avg_pnl_pct': None}
                continue
            wins = sum(1 for t in trades if (t.get('pnl_pct') or 0) > 0)
            avg = sum((t.get('pnl_pct') or 0) for t in trades) / n
            result[name] = {
                'count': n,
                'win_rate': round(wins / n * 100, 1),
                'avg_pnl_pct': round(avg, 2),
            }
        return result
    
    def manual_close(self, symbol: str, reason: str = 'manual') -> Dict:
        """Close a real position. Default reason 'manual' (UI button); callers
        like the WAIT-gate pass their own reason so the close is labelled
        correctly instead of always showing 'Manual'."""
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return {'ok': False, 'reason': 'No open position for symbol'}
        current = self._get_current_price(symbol) or pos['entry_price']
        self._close_position(symbol, current, reason=reason)
        return {'ok': True}
    
    def manual_close_shadow(self, symbol: str, reason: str = 'manual') -> Dict:
        """Close a paper position. See manual_close re: reason."""
        with self._lock:
            pos = self._shadow_positions.get(symbol)
        if not pos:
            return {'ok': False, 'reason': 'No open paper position for symbol'}
        current = self._get_current_price(symbol) or pos['entry_price']
        self._close_shadow(symbol, current, reason=reason)
        return {'ok': True}
    
    def close_all_with_queue(self, reason: str = 'auto_gate_wait',
                              throttle_secs: float = 0.4) -> Dict:
        """Close EVERY open position — real (💼 Trade Manager) and paper
        (🧪 Test Mode) — one at a time through a throttled queue.

        Why a queue (2026-06-14): firing N close orders at the exchange
        simultaneously risks rate-limit rejections and partially-filled
        closes, which can leave a position half-open on Bybit. Closing
        sequentially with a small gap lets each reduce-only order confirm
        before the next, so the exchange isn't overwhelmed and every
        position fully closes. Paper positions don't hit the exchange but
        go through the same queue for consistent ordering/logging.

        Returns {ok, closed_real, closed_shadow, failed}.
        """
        import time as _t
        # Snapshot the symbol lists under lock, then release so each close
        # (which re-acquires the lock) doesn't deadlock.
        with self._lock:
            real_syms = list(self._positions.keys())
            shadow_syms = list(self._shadow_positions.keys())
        closed_real, closed_shadow, failed = [], [], []

        for sym in real_syms:
            try:
                r = self.manual_close(sym, reason=reason)
                if r.get('ok'):
                    closed_real.append(sym)
                else:
                    failed.append({'symbol': sym, 'kind': 'real',
                                   'reason': r.get('reason', 'unknown')})
            except Exception as e:
                failed.append({'symbol': sym, 'kind': 'real', 'reason': str(e)})
            _t.sleep(throttle_secs)     # let the exchange confirm before next

        for sym in shadow_syms:
            try:
                r = self.manual_close_shadow(sym, reason=reason)
                if r.get('ok'):
                    closed_shadow.append(sym)
                else:
                    failed.append({'symbol': sym, 'kind': 'shadow',
                                   'reason': r.get('reason', 'unknown')})
            except Exception as e:
                failed.append({'symbol': sym, 'kind': 'shadow', 'reason': str(e)})
            _t.sleep(throttle_secs)

        if closed_real or closed_shadow:
            print(f"[TM] close_all_with_queue ({reason}): "
                  f"real={len(closed_real)} shadow={len(closed_shadow)} "
                  f"failed={len(failed)}")
        return {'ok': True, 'closed_real': closed_real,
                'closed_shadow': closed_shadow, 'failed': failed,
                'reason': reason}
    
    @staticmethod
    def _record_manual_hist(pos: Dict, field: str, value):
        """Append a Manual SL/TP level to the position's change-history so the
        chronology can show ALL levels that ever applied during the trade
        (not just the last one). field='sl'|'tp'. Only real set-values are kept
        (clears are simply not appended); consecutive duplicates are skipped."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if v <= 0:
            return
        key = 'manual_sl_hist' if field == 'sl' else 'manual_tp_hist'
        hist = pos.get(key)
        if not isinstance(hist, list):
            hist = []
            pos[key] = hist
        v = round(v, 10)
        if not hist or hist[-1] != v:
            hist.append(v)

    def _fmt_manual_hist(self, pos: Dict) -> str:
        """Compact «Manual SL: a → b · Manual TP: c» summary of ALL levels that
        applied during the trade — for the close chronology entry. Ensures the
        CURRENT value is included even if a setter path missed the history."""
        def _seq(histkey, curkey):
            seq = list(pos.get(histkey) or [])
            cur = pos.get(curkey)
            try:
                cur = float(cur) if cur is not None else None
            except (TypeError, ValueError):
                cur = None
            if cur and cur > 0:
                cur = round(cur, 10)
                if not seq or seq[-1] != cur:
                    seq.append(cur)
            return seq
        parts = []
        sls = _seq('manual_sl_hist', 'manual_sl')
        tps = _seq('manual_tp_hist', 'manual_tp')
        if sls:
            parts.append('Manual SL: ' + ' → '.join(self._fmt_sltp(v) for v in sls))
        if tps:
            parts.append('Manual TP: ' + ' → '.join(self._fmt_sltp(v) for v in tps))
        return ' · '.join(parts)

    def update_manual_sl_tp(self, symbol: str, manual_sl=None,
                              manual_tp=None, is_shadow: bool = False) -> Dict:
        """Set or clear the per-position manual SL/TP override.
        
        `manual_sl` / `manual_tp` semantics:
          - None  → leave field unchanged (this is how the UI can update
                    one without touching the other)
          - 0 / '' → CLEAR the field (no manual stop on that side)
          - >0  → set to that absolute price level
        
        VALIDATION (added 2026-06-08, fix for XAGUSDT premature close):
        For 'set' operations, the level must be on the correct side of
        the CURRENT market price for the position's direction. Otherwise
        the next monitor tick would trigger an instant exit because the
        "trigger condition" is already true.
        
            LONG  · manual_sl must be < current_price (stop fires on drop)
            LONG  · manual_tp must be > current_price (take fires on rise)
            SHORT · manual_sl must be > current_price (stop fires on rise)
            SHORT · manual_tp must be < current_price (take fires on drop)
        
        We validate against CURRENT price, not entry, because the user may
        want to lock in profit after the position has moved favorably (e.g.
        SHORT opened at 75, price now at 67 — SL=70 is valid: it's above
        current 67, below entry 75, locking partial profit if price rises).
        
        On validation failure, returns {ok: false, reason: '...',
        validation: true}, and NOTHING is mutated. The frontend uses
        `validation: true` to alert the user (vs the benign "position
        already closed" reason which is silent).
        
        Returns ok plus the updated position dict (with normalized fields).
        Persists immediately so a server restart preserves the override.
        """
        def _parse(v):
            """Returns ('skip',), ('clear',) or ('set', float)."""
            if v is None:
                return ('skip',)
            try:
                f = float(v)
            except (TypeError, ValueError):
                return ('skip',)
            if f <= 0:
                return ('clear',)
            return ('set', f)
        
        sl_op = _parse(manual_sl)
        tp_op = _parse(manual_tp)
        
        store = self._shadow_positions if is_shadow else self._positions
        kind = 'shadow' if is_shadow else 'real'
        
        with self._lock:
            pos = store.get(symbol)
            if not pos:
                return {'ok': False,
                        'reason': f'No open {kind} position for {symbol}'}
            
            # === Directional validation ===
            # Only kicks in when at least one operation is 'set'. Skip and
            # clear operations are always safe. We snapshot the side and
            # entry price under the lock, then release the lock before
            # the price lookup (which may hit the network in the worst
            # case — we don't want to block other position monitors).
            side = pos.get('side')
            entry_price = pos.get('entry_price') or 0.0
        
        need_validation = (sl_op[0] == 'set') or (tp_op[0] == 'set')
        if need_validation:
            # Use the live monitor's price source; fall back to entry if
            # market data isn't available right this instant. Falling back
            # to entry is a reasonable degraded mode — the user's level
            # will still be sane vs entry even if not vs current.
            current_price = self._get_current_price(symbol) or entry_price
            if current_price <= 0:
                return {'ok': False,
                        'reason': f'Cannot validate: no price available for {symbol}',
                        'validation': True}
            
            errors = []
            if sl_op[0] == 'set':
                level = sl_op[1]
                if side == 'LONG' and not (level < current_price):
                    errors.append(
                        f"SL ${level} must be BELOW current price ${current_price:.6g} "
                        f"for LONG (stop fires when price drops)")
                elif side == 'SHORT' and not (level > current_price):
                    errors.append(
                        f"SL ${level} must be ABOVE current price ${current_price:.6g} "
                        f"for SHORT (stop fires when price rises)")
            if tp_op[0] == 'set':
                level = tp_op[1]
                if side == 'LONG' and not (level > current_price):
                    errors.append(
                        f"TP ${level} must be ABOVE current price ${current_price:.6g} "
                        f"for LONG (take fires when price rises)")
                elif side == 'SHORT' and not (level < current_price):
                    errors.append(
                        f"TP ${level} must be BELOW current price ${current_price:.6g} "
                        f"for SHORT (take fires when price drops)")
            
            if errors:
                # Reject atomically — nothing gets applied. User must
                # correct the input. This is the fix that prevents the
                # XAGUSDT-style premature close.
                msg = 'Invalid SL/TP for ' + side + ' ' + symbol + ': ' + '; '.join(errors)
                print(f"[TM] 🚫 Manual SL/TP rejected for {symbol} ({kind}): {msg}")
                return {'ok': False, 'reason': msg, 'validation': True}
        
        # All validations passed (or none were needed) — apply now.
        with self._lock:
            pos = store.get(symbol)
            if not pos:
                # Position closed between our checks and now. Edge race
                # but possible. Bail out with benign reason.
                return {'ok': False,
                        'reason': f'No open {kind} position for {symbol}'}
            
            if sl_op[0] == 'set':
                _sv = self._round_sltp_value(sl_op[1])
                pos['manual_sl'] = _sv
                self._record_manual_hist(pos, 'sl', _sv)
            elif sl_op[0] == 'clear':
                pos.pop('manual_sl', None)

            if tp_op[0] == 'set':
                _tv = self._round_sltp_value(tp_op[1])
                pos['manual_tp'] = _tv
                self._record_manual_hist(pos, 'tp', _tv)
            elif tp_op[0] == 'clear':
                pos.pop('manual_tp', None)

            updated = dict(pos)
        
        # Persist outside the lock — DB call can be slow on Render with
        # PostgreSQL backend, no need to block monitors.
        try:
            if is_shadow:
                self._persist_shadow_positions()
            else:
                self._persist_positions()
        except Exception as e:
            print(f"[TM] manual SL/TP persist warn for {symbol}: {e}")
        
        # Log the change so operator can correlate with later closes.
        sl_disp = f"{updated.get('manual_sl')}" if 'manual_sl' in updated else '—'
        tp_disp = f"{updated.get('manual_tp')}" if 'manual_tp' in updated else '—'
        print(f"[TM] Manual SL/TP updated for {symbol} ({kind}): "
              f"SL={sl_disp} TP={tp_disp}")
        return {'ok': True, 'position': updated}
    
    def update_manual_mode(self, symbol: str, enabled: bool,
                            is_shadow: bool = False) -> Dict:
        """Toggle per-position MANUAL MODE.
        
        When manual_mode=True for a position:
          - Automatic exits (SL, TP, time stop, HTF flip, Reverse SMC,
            CHoCH/BOS triggers, trailing updates, BE updates) are
            BYPASSED. The only things that can close the position are:
              1. Manual SL / Manual TP per-position price levels
              2. The user clicking "Close" in the UI (force close)
          - New SMC signals on this symbol are IGNORED — no new open,
            no reverse. The user has taken full control of this trade.
        
        When manual_mode=False (default), all automatic logic runs as
        usual — current strategy behavior is unchanged. This is purely
        an opt-in escape hatch for trades the user wants to manage by
        hand without disabling TM globally.
        
        Works for both real and shadow positions — only `is_shadow`
        selects which store.
        """
        store = self._shadow_positions if is_shadow else self._positions
        kind = 'shadow' if is_shadow else 'real'
        
        with self._lock:
            pos = store.get(symbol)
            if not pos:
                return {'ok': False,
                        'reason': f'No open {kind} position for {symbol}'}
            pos['manual_mode'] = bool(enabled)
            updated = dict(pos)
        
        # Persist outside the lock
        try:
            if is_shadow:
                self._persist_shadow_positions()
            else:
                self._persist_positions()
        except Exception as e:
            print(f"[TM] manual mode persist warn for {symbol}: {e}")
        
        state = 'ON' if enabled else 'OFF'
        print(f"[TM] Manual mode {state} for {symbol} ({kind}): "
              f"new signals/auto-exits {'ignored' if enabled else 'active'}")
        return {'ok': True, 'position': updated, 'manual_mode': bool(enabled)}
    
    def manual_open(self, symbol: str, side: str, bypass_gates: bool = False,
                    opened_by: Optional[str] = None) -> Dict:
        """User-initiated position open from the Decision Center panel.

        Uses Position Sizing settings from TM (sizing_mode, fixed_usd_amount,
        leverage, use_sl, use_tp, sl_pct, tp_pct etc.) — exactly the same
        path as auto-opened positions, but triggered by an explicit user
        click rather than an SMC signal.

        bypass_gates: If True, ignore LONG/SHORT entry gates (used by Fuel Auto-Filter).

        Validations before placing the order:
          - TM must be enabled (master toggle ON)
          - Symbol must not already have an open position (real or shadow)
            in TM — would cause confusing state
          - side ∈ {'LONG', 'SHORT'}
          - Bybit must be configured (API key)
          - Current price must be available (market_data lookup)

        Returns:
          { ok: bool, reason?: str, position?: {...} }
        """
        symbol = (symbol or '').upper().strip()
        if not symbol:
            return {'ok': False, 'reason': 'symbol required'}
        side = (side or '').upper().strip()
        if side not in ('LONG', 'SHORT'):
            return {'ok': False, 'reason': f"side must be LONG or SHORT, got {side!r}"}

        # === Fuel Auto-Filter interception ===
        # A manual LONG/SHORT click is also queued in the ❤️ FF base while FF is
        # on (bypass_gates=True means this IS FF opening — don't re-intercept).
        if not bypass_gates:
            try:
                from detection.fuel_filter import get_fuel_filter
                ff = get_fuel_filter()
                if ff and ff.is_enabled():
                    _disp = ff.intercept(symbol, side)
                    if _disp == 'queued':
                        return {'ok': True, 'queued': True,
                                'reason': f'{symbol} {side} → черга ❤️ Fuel Auto-Filter '
                                          f'(чекає фільтр черги)'}
                    if _disp == 'dropped':
                        return {'ok': False,
                                'reason': f'{symbol} {side} відкинуто Чергою-2 '
                                          f'(CTR-нахил не в бік сигналу)'}
                    # '' → both queues off → open directly below.
            except Exception as e:
                print(f"[TM] FF intercept error (manual) for {symbol}: {e}")

        test_mode = self._settings.get('test_mode', True)

        # Check max positions limit
        at_limit = self._at_max_positions()
        if at_limit and not test_mode:
            return {'ok': False, 'reason':
                    f'Max open positions ({self._settings.get("max_open_positions", 10)}) '
                    f'reached. Enable Test Mode to open additional paper positions.'}

        with self._lock:
            if symbol in self._positions:
                return {'ok': False, 'reason':
                        f'Already have an open position for {symbol}. '
                        f'Close it first or use Manual SL/TP to manage.'}
            if symbol in self._shadow_positions:
                return {'ok': False, 'reason':
                        f'Symbol {symbol} has a paper (shadow) position. '
                        f'Close it first to avoid mixed real/paper state.'}

        # Fetch fresh price
        entry_price = self._get_current_price(symbol)
        if not entry_price or entry_price <= 0:
            return {'ok': False, 'reason':
                    f'Could not fetch current price for {symbol}'}

        # Decision: real or paper?
        # If TM enabled AND not at limit: open real.
        # Otherwise if test_mode on: open shadow.
        # NOTE: bypass_gates does NOT bypass max_open_positions — the limit is
        # a hard ceiling for REAL positions. bypass_gates only affects the
        # LONG/SHORT entry gates inside _open_position (quality/trend checks).
        open_real = False
        if self.is_enabled() and not at_limit:
            if not self.bybit or not getattr(self.bybit, 'api_key', None):
                return {'ok': False, 'reason': 'Bybit not configured (no API key)'}
            open_real = True

        if open_real:
            try:
                res = self._open_position(symbol, side, entry_price,
                                          opened_by=(opened_by or 'manual_ui'),
                                          bypass_gates=bypass_gates)
            except Exception as e:
                return {'ok': False, 'reason': f'Open error: {e}'}
            with self._lock:
                pos = self._positions.get(symbol)
            if not pos:
                # Surface the EXACT reason _open_position reported (gate /
                # sizing / exchange rejection) instead of a generic message.
                reason = (res or {}).get('reason') if isinstance(res, dict) else None
                return {'ok': False, 'reason':
                        reason or 'Order placement failed — check Telegram/logs for details'}
            return {'ok': True, 'position': dict(pos), 'entry_price': entry_price,
                    'mode': 'real'}
        elif test_mode:
            # At limit, but test_mode is on → open shadow.
            # Pass bypass_gates so FF force-opens (and any explicit manual
            # bypass) skip the directional + fuel-confirm gates on the paper
            # path too, matching the real path above.
            try:
                self._open_shadow(symbol, side, entry_price,
                                  opened_by=(opened_by or 'manual_ui_overflow'),
                                  bypass_gates=bypass_gates)
            except Exception as e:
                return {'ok': False, 'reason': f'Shadow open error: {e}'}
            with self._lock:
                pos = self._shadow_positions.get(symbol)
            if not pos:
                return {'ok': False, 'reason': 'Shadow position creation failed'}
            print(f"[TM] ℹ️ Max positions reached ({self._settings.get('max_open_positions')}), "
                  f"opened {symbol} in Test Mode instead")
            return {'ok': True, 'position': dict(pos), 'entry_price': entry_price,
                    'mode': 'test', 'reason': 'Max positions limit reached'}
        else:
            return {'ok': False, 'reason':
                    f'TM disabled and max positions reached. Enable Test Mode or TM.'}
    
    @staticmethod
    def _pop_closed_match(lst: list, match: Dict):
        """Remove ONE record matching (symbol, closed_at, exit_price),
        searching from the end (newest first). Returns the record or None.
        
        Why match-based (2026-06-12): get_state() ships only the LAST 50
        closes to the UI, while index-based deletion popped from the FULL
        list — with >50 records the indexes diverged and deletes hit the
        wrong (hidden, old) entry, so visible rows "refused to delete".
        A (symbol, closed_at, exit_price) triple identifies a close
        unambiguously regardless of list length or slicing.
        """
        sym = match.get('symbol')
        cat = match.get('closed_at')
        xp = match.get('exit_price')
        for i in range(len(lst) - 1, -1, -1):
            c = lst[i]
            if (c.get('symbol') == sym
                    and c.get('closed_at') == cat
                    and (xp is None or c.get('exit_price') == xp)):
                return lst.pop(i)
        return None
    
    def delete_closed_trade(self, idx=None, match: Dict = None) -> Dict:
        """Permanently remove a closed real trade — by match (preferred,
        see _pop_closed_match) or legacy index. Stats are recomputed on
        the fly inside get_state(), so removing the entry from the list
        is enough — the next state poll will show updated PnL/win rate
        as if the trade never happened.
        """
        with self._lock:
            if match:
                removed = self._pop_closed_match(self._closed_trades, match)
                if removed is None:
                    return {'ok': False, 'reason': 'record not found'}
            else:
                try:
                    idx = int(idx)
                except Exception:
                    return {'ok': False, 'reason': 'invalid index'}
                if idx < 0 or idx >= len(self._closed_trades):
                    return {'ok': False, 'reason': 'index out of range'}
                removed = self._closed_trades.pop(idx)
        self._persist_closed_trades()
        return {'ok': True, 'removed': removed.get('symbol', '')}
    
    def delete_shadow_closed_trade(self, idx=None, match: Dict = None) -> Dict:
        """Permanently remove a closed paper trade — by match (preferred)
        or legacy index (see _pop_closed_match for the why)."""
        with self._lock:
            if match:
                removed = self._pop_closed_match(self._shadow_closed, match)
                if removed is None:
                    return {'ok': False, 'reason': 'record not found'}
            else:
                try:
                    idx = int(idx)
                except Exception:
                    return {'ok': False, 'reason': 'invalid index'}
                if idx < 0 or idx >= len(self._shadow_closed):
                    return {'ok': False, 'reason': 'index out of range'}
                removed = self._shadow_closed.pop(idx)
        self._persist_shadow_closed()
        return {'ok': True, 'removed': removed.get('symbol', '')}
    
    def clear_shadow_closed(self) -> Dict:
        """Wipe the entire Recent Paper Closes history. Stats recompute
        from the (now empty) list on the next state poll."""
        with self._lock:
            n = len(self._shadow_closed)
            self._shadow_closed = []
        self._persist_shadow_closed()
        return {'ok': True, 'cleared': n}

    def clear_closed(self) -> Dict:
        """Wipe the entire Recent Closed Trades (real) history. Stats
        recompute from the (now empty) list on the next state poll."""
        with self._lock:
            n = len(self._closed_trades)
            self._closed_trades = []
        self._persist_closed_trades()
        return {'ok': True, 'cleared': n}
    
    # ============================================================
    # Notifications
    # ============================================================
    
    def _notify(self, msg: str, is_test: bool = False, category: Optional[str] = None):
        """Send Telegram notification, respecting the relevant toggle.

        Args:
            msg: text to send
            is_test: True if from shadow/paper trade — gated by test_telegram_alerts.
                     False if from real position — gated by telegram_alerts.
            category: when set (e.g. 'trades'), ALSO mirror the message to that
                     Telegram GROUP category/topic via the bot's notify_category
                     (📈 Угоди тема). Best-effort; never blocks the main send.
        """
        # Mirror to the group category/topic first (independent of the private
        # notifier toggle) so trade events land in the 📈 Угоди тема.
        if category:
            try:
                from web.tg_bot import notify_category
                notify_category(category, msg)
            except Exception as e:
                print(f"[TM] category notify error: {e}")
        if not self.notifier:
            return
        # Gate by toggle. Default both to True if missing (back-compat with
        # earlier saved settings that don't have these keys).
        if is_test:
            if not self._settings.get('test_telegram_alerts', True):
                return
        else:
            if not self._settings.get('telegram_alerts', True):
                return
        # Tag the PRIVATE copy too so the bot's private chat is filterable by
        # category (a private chat has no topics — the hashtag is the category).
        priv = msg
        if category:
            try:
                from web.tg_bot import cat_tag
                _t = cat_tag(category)
                if _t:
                    priv = f"{_t}\n{msg}"
            except Exception:
                pass
        try:
            self.notifier.send_message(priv)
        except Exception as e:
            print(f"[TM] Notify error: {e}")
    
    def _notify_open(self, pos):
        side = pos['side']
        icon = '🟢' if side == 'LONG' else '🔴'
        sl_str = self._fmt_price(pos['sl_price']) if pos.get('sl_price') else '—'
        tp_str = self._fmt_price(pos['tp_price']) if pos.get('tp_price') else '—'
        # Entry Score block — same renderer as shadow side for consistency
        es_block = self._format_entry_score_telegram(pos.get('entry_score'))
        # Full opened_by (Forecast 1H + CTR + Entry score snapshot)
        src_line = ''
        if pos.get('opened_by'):
            src_line = f"📋 Джерело: {self._base_opened_by(pos['opened_by'])}\n"
        msg = (
            f"{icon} ▶️ <b>ВІДКРИТО {side}</b> · #{pos['symbol']}\n"
            f"━━━━━━━━━━━━\n"
            f"📍 Вхід: {self._fmt_price(pos['entry_price'])}\n"
            f"💼 К-ть: {pos['qty']:.6g} · ⚙️ Плече: {self._settings.get('leverage', 10)}x\n"
            f"🛡 SL: {sl_str} · 🎯 TP: {tp_str}\n"
            f"{es_block}"
            f"{src_line}"
        ).rstrip()
        self._notify(msg, category='trades')

    def _notify_close(self, closed):
        side = closed['side']
        pnl_pct = closed['pnl_pct']
        pnl_usd = closed['pnl_usd']
        is_win = pnl_pct > 0
        icon = '✅' if is_win else '❌'
        # Show what the entry score said when we opened — useful for
        # weight-tuning and spotting when the predictor was wrong.
        es_recap = self._format_entry_score_recap(closed.get('entry_score'))
        peak_line = self._fmt_peak_line(closed)
        msg = (
            f"{icon} ⏹ <b>ЗАКРИТО {side}</b> {pnl_pct:+.2f}% ({pnl_usd:+.2f}$) · #{closed['symbol']}\n"
            f"━━━━━━━━━━━━\n"
            f"📍 Вхід → Вихід: {self._fmt_price(closed['entry_price'])} → {self._fmt_price(closed['exit_price'])}\n"
            f"🔖 Причина: {self._reason_label(closed['reason'])}\n"
            f"{peak_line}"
            f"{es_recap}"
        ).rstrip()
        self._notify(msg, category='trades')

    def _fmt_peak_line(self, closed) -> str:
        """📈 «Пік» line for a close notification — the max favourable value the
        trade reached (peak PnL % + the price at that peak). '' when unknown."""
        peak = closed.get('peak_pnl_pct')
        if not isinstance(peak, (int, float)):
            return ''
        pps = ''
        try:
            ep = float(closed.get('entry_price') or 0)
            if ep > 0:
                pp = ep * (1 + peak / 100.0) if closed.get('side') == 'LONG' \
                    else ep * (1 - peak / 100.0)
                pps = f" (@ {self._fmt_price(pp)})"
        except Exception:
            pps = ''
        return f"📈 Пік: {peak:+.2f}%{pps}\n"

    @staticmethod
    def _reason_label(reason: str) -> str:
        return {
            'stop_loss': '🛡 Stop Loss',
            'take_profit': '🎯 Take Profit',
            'reverse_smc': '🔄 Reverse SMC (CHoCH)',
            'reverse_signal': '🔁 Reverse Signal (qualified)',
            'opposite_ob_exit': '🔃 Opposite OB Exit',
            'forecast_1h_confluence': '🔮 Forecast 1H Confluence',
            'htf_flip': '📡 HTF Trend Flip',
            'time_stop': '⏱ Time Stop',
            'trailing_stop': '📈 Trailing Stop',
            'bot_sl': '🤖🛡 Бот-софт SL',
            'bot_tp_trail': '🤖📈 Бот-софт трейл-TP',
            'ctr_reversal_after_peak': '🔄 Розворот після піку',
            'manual': '✋ Manual',
            'manual_sl': '✋🛡 Manual SL',
            'manual_tp': '✋🎯 Manual TP',
            'external_close': '🔄 External Close (off-bot)',
            'mm_below_min': '📉 ММ нижче порога',
            'btc_flip': '₿ Розворот банера',
            'fuel_flipped': '🔄 ММ розворот',
            'fuel_faded': '💨 ММ згасло',
            'potential_reached': '🔥 Виснаженість',
            'wait_verdict': '⏸ WAIT',
            'bos_2_partial': '✂️ BOS-2 partial',
            'bos_3_partial': '✂️ BOS-3 partial',
            'bos_4_partial': '✂️ BOS-4 partial',
        }.get(reason, reason)
    
    @staticmethod
    def _fmt_price(price: float) -> str:
        if price <= 0:
            return '$0'
        if price < 0.0001:
            return f"${price:.8f}"
        if price < 0.01:
            return f"${price:.6f}"
        if price < 1:
            return f"${price:.5f}"
        if price < 100:
            return f"${price:.4f}"
        return f"${price:,.2f}"

    @staticmethod
    def _fmt_sltp(price) -> str:
        """Manual SL/TP price formatter: 2 decimals for ≥$1 (kills float tails
        like 190.0836000…02), more precision for sub-$1 so the level survives
        (trailing zeros stripped)."""
        try:
            n = float(price)
        except (TypeError, ValueError):
            return '—'
        if n <= 0:
            return '—'
        if n >= 1:
            return f"${n:,.2f}"
        dp = 8 if n < 0.0001 else (6 if n < 0.01 else 5)
        return '$' + (f"%.{dp}f" % n).rstrip('0').rstrip('.')

    @staticmethod
    def _round_sltp_value(price):
        """Round a Manual SL/TP price to the stored precision (2dp for ≥$1, more
        for sub-$1) — kills float tails (190.0836…02) at the SOURCE so every
        consumer sees a clean value. Returns the input unchanged on bad input."""
        try:
            n = float(price)
        except (TypeError, ValueError):
            return price
        if n <= 0:
            return n
        if n >= 1:
            return round(n, 2)
        dp = 8 if n < 0.0001 else (6 if n < 0.01 else 5)
        return round(n, dp)


# Singleton
_instance: Optional[TradeManager] = None


def get_trade_manager() -> Optional[TradeManager]:
    return _instance


def init_trade_manager(db=None, notifier=None, bybit=None, scanner=None) -> TradeManager:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = TradeManager(db=db, notifier=notifier, bybit=bybit, scanner=scanner)
    return _instance
