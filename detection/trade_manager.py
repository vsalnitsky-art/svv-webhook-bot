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
CLOSED_TRADES_LIMIT = 100   # keep this many recent closed trades
INITIAL_DELAY_SECS = 20      # wait at startup before first tick

DEFAULT_SETTINGS = {
    # Master toggle — DEFAULT OFF for safety
    'enabled': False,
    
    # === Position Sizing ===
    'sizing_mode': 'fixed_usd',     # 'fixed_usd' | 'fixed_pct' | 'risk_based'
    'fixed_usd_amount': 100.0,       # USD per trade (mode = fixed_usd)
    'fixed_pct_balance': 2.0,        # % of balance (mode = fixed_pct)
    'risk_pct_balance': 1.0,         # max loss as % of balance (mode = risk_based)
    'leverage': 10,                  # 1-50 typically
    
    # === Exit Rules — each toggleable ===
    'use_sl': True,                  # Stop Loss
    'sl_pct': 2.0,                   # %
    
    'use_tp': True,                  # Take Profit
    'tp_pct': 5.0,                   # %
    
    'use_reverse_smc': True,         # Close on opposite CHoCH
    
    'use_htf_flip': True,            # Close when HTF trend flips against us
    
    'use_time_stop': False,          # Close after N hours regardless
    'time_stop_hours': 4,
    
    'use_trailing': True,            # Trailing stop
    'trailing_activate_pct': 1.0,    # activate after +X% profit
    'trailing_distance_pct': 0.5,    # trail by Y% from peak
    
    'use_be': True,                  # Break-Even Move
    'be_trigger_pct': 0.5,           # move SL to entry after +X% profit
    
    # === Forecast 1H Confluence Close exit ===
    # Closes position when both:
    #   1. Opposite CHoCH appears on LTF (after position open)
    #   2. Forecast 1H side is opposite to position
    # Either condition alone won't close — needs both (confluence).
    'use_forecast_1h_close': True,
    
    # === Test mode (paper trading for exit-rule validation) ===
    # When ON: signals create "shadow" positions tracked in memory only.
    # No Bybit orders. Exit rules still evaluate and send Telegram alerts
    # so the user can validate strategy behavior without real risk.
    # Real positions (when TM enabled=True) take precedence over shadow.
    'test_mode': True,
    
    # === Telegram notification toggles ===
    # Independently control Telegram alerts for real positions vs paper trades.
    # SMC scanner always sends its own signal alerts — these toggles control
    # the EXTRA alerts from Trade Manager about position lifecycle.
    'telegram_alerts': True,        # real position open/close/partial
    'test_telegram_alerts': True,   # paper [TEST] open/close
    
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
        'weight_forecast_alignment': 30.0,
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
        'weight_forecast_alignment': 25.0,
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
        
        self._load_positions()
        self._load_closed_trades()
        self._load_shadow_positions()
        self._load_shadow_closed()
        
        # Initialize evaluator state for any positions we restored from DB.
        # We don't have history for them, so peak = current PnL at first
        # tick (will be patched in _monitor_position), counts start at 0.
        for sym in self._positions:
            self._pos_state[sym] = self._fresh_pos_state()
        for sym in self._shadow_positions:
            self._shadow_pos_state[sym] = self._fresh_pos_state()
        
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
            
            for k in ['fixed_usd_amount', 'fixed_pct_balance', 'risk_pct_balance',
                      'sl_pct', 'tp_pct', 'time_stop_hours',
                      'trailing_activate_pct', 'trailing_distance_pct', 'be_trigger_pct',
                      'bos_2_close_pct', 'bos_3_close_pct', 'bos_4_close_pct']:
                try:
                    self._settings[k] = float(self._settings.get(k, DEFAULT_SETTINGS.get(k, 0)))
                except:
                    self._settings[k] = float(DEFAULT_SETTINGS.get(k, 0))
            
            for k in ['use_sl', 'use_tp', 'use_reverse_smc', 'use_htf_flip',
                      'use_time_stop', 'use_trailing', 'use_be',
                      'use_forecast_1h_close', 'test_mode',
                      'telegram_alerts', 'test_telegram_alerts',
                      'use_bos_partials', 'trailing_after_bos_2',
                      'health_score_enabled',
                      'entry_score_enabled']:
                self._settings[k] = bool(self._settings.get(k, False))
            
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
        with self._lock:
            symbols = list(self._positions.keys())
        
        for sym in symbols:
            try:
                self._monitor_position(sym)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[TM] Monitor error {sym}: {e}")
                self._errors += 1
    
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
    
    def _update_trailing(self, pos, current_price):
        s = self._settings
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
                self._close_position(pos['symbol'], current_price, reason='trailing_stop')
        else:
            trail_stop = pos['trailing_peak'] * (1 + dist_pct)
            if current_price >= trail_stop:
                self._close_position(pos['symbol'], current_price, reason='trailing_stop')
    
    def _update_be(self, pos, current_price):
        s = self._settings
        if not s.get('use_be') or pos.get('be_moved'):
            return
        
        trigger = s.get('be_trigger_pct', 0.5) / 100
        entry = pos['entry_price']
        if pos['side'] == 'LONG' and current_price >= entry * (1 + trigger):
            pos['sl_price'] = entry
            pos['be_moved'] = True
            self._update_exchange_sl(pos['symbol'], entry)
            self._notify(f"⚖️ BE: SL → entry for {pos['symbol']} @ {self._fmt_price(entry)}")
            self._persist_positions()
        elif pos['side'] == 'SHORT' and current_price <= entry * (1 - trigger):
            pos['sl_price'] = entry
            pos['be_moved'] = True
            self._update_exchange_sl(pos['symbol'], entry)
            self._notify(f"⚖️ BE: SL → entry for {pos['symbol']} @ {self._fmt_price(entry)}")
            self._persist_positions()
    
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
        """
        s = self._settings
        enabled = self.is_enabled()
        test_mode = s.get('test_mode', True)
        
        with self._lock:
            existing_real = self._positions.get(symbol)
            existing_shadow = self._shadow_positions.get(symbol)
        
        if enabled:
            # Real-money mode
            if existing_real:
                if existing_real['side'] == side:
                    # Same direction — already in this trend, no-op
                    return
                # OPPOSITE direction — reverse: close + open
                print(f"[TM] 🔄 Reverse signal for {symbol}: "
                      f"closing {existing_real['side']} → opening {side}")
                try:
                    self._close_position(symbol, entry_price, reason='reverse_signal')
                except Exception as e:
                    print(f"[TM] ❌ Reverse-close failed for {symbol}: {e}")
                    # Bail out — don't leave the user with two positions or
                    # one stale position if the close call errored. Better
                    # to skip the new open and let next signal retry.
                    return
                if not self._is_tradeable(symbol):
                    print(f"[TM] {symbol} not in tradeable list — reverse-open skipped")
                    return
                self._open_position(symbol, side, entry_price, opened_by)
                return
            # No existing real position
            if not self._is_tradeable(symbol):
                print(f"[TM] {symbol} not in tradeable list — signal ignored")
                return
            self._open_position(symbol, side, entry_price, opened_by)
        elif test_mode:
            # Paper mode — track shadow position
            if existing_shadow:
                if existing_shadow['side'] == side:
                    return
                # OPPOSITE direction — same reverse semantics for paper trades
                print(f"[TM] 🔄 [TEST] Reverse signal for {symbol}: "
                      f"closing {existing_shadow['side']} → opening {side}")
                try:
                    self._close_shadow(symbol, entry_price, reason='reverse_signal')
                except Exception as e:
                    print(f"[TM] ❌ [TEST] Reverse-close-shadow failed for {symbol}: {e}")
                    return
                self._open_shadow(symbol, side, entry_price, opened_by)
                return
            self._open_shadow(symbol, side, entry_price, opened_by)
    
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
                    elif shadow and s.get('test_mode'):
                        self._close_shadow(symbol, current_price,
                                            reason='forecast_1h_confluence')
                    return  # closed; don't fall through to plain reverse
        
        # === Plain Reverse SMC (CHoCH only) ===
        if s.get('use_reverse_smc'):
            current_price = self._get_current_price(symbol) or pos['entry_price']
            if real:
                self._close_position(symbol, current_price, reason='reverse_smc')
            elif shadow and s.get('test_mode'):
                self._close_shadow(symbol, current_price, reason='reverse_smc')
    
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
        
        Format: '<base> · 🧠 LONG 78% (good)'
        
        Previously this surfaced Forecast/CTR/Entry as three separate
        chunks but the user feedback was that it was hard to read. Now
        we delegate the whole verdict to the Decision Center headline,
        which is one phrase humans actually parse at a glance. The base
        opened_by tag (e.g. 'choch_bos', 'choch') is preserved as the
        prefix so existing logic that inspects it keeps working.
        
        Falls back to '<base>' alone when no decision is available.
        """
        parts = [opened_by]
        if entry_score:
            headline = entry_score.get('headline')
            verdict = entry_score.get('verdict', '')
            if headline:
                parts.append(f"🧠 {headline} ({verdict})" if verdict else f"🧠 {headline}")
            else:
                # Legacy entry_score shape — score+verdict only
                score = entry_score.get('score')
                if score is not None:
                    sign = '+' if score >= 0 else ''
                    if verdict:
                        parts.append(f"🧠 {sign}{score:.0f} ({verdict})")
                    else:
                        parts.append(f"🧠 {sign}{score:.0f}")
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
        ctx['forecast'] = self._get_forecast_1h(symbol)
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
        # Two independent code paths:
        #   1) Real positions — gated by is_enabled() AND use_bos_partials,
        #      operate on _positions and call Bybit API via _partial_close
        #   2) Shadow positions — gated by use_bos_partials only (test_mode
        #      is supposed to work with TM master toggle OFF), operate on
        #      _shadow_positions, no Bybit calls.
        # Both paths run on every BOS event so a symbol with both a real
        # AND a shadow position would partial-close both. In practice
        # users only have one of the two at a time but the code is robust
        # to either configuration.
        
        if not s.get('use_bos_partials'):
            # User explicitly disabled the feature — log once and skip both paths.
            print(f"[TM] BOS {symbol} {direction} ignored: use_bos_partials=False")
            return
        
        # ----- REAL position path -----
        if real and self.is_enabled():
            self._process_bos_real(symbol, direction, level, bar_t, real)
        elif real and not self.is_enabled():
            print(f"[TM] BOS {symbol} {direction} for real position ignored: TM disabled")
        
        # ----- SHADOW position path -----
        # Always runs when a shadow position exists. test_mode by design
        # operates while TM master toggle is off.
        if shadow:
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
    
    def _open_position(self, symbol: str, side: str, entry_price: float, opened_by: str):
        s = self._settings
        
        # Calculate size
        try:
            qty = self._calculate_qty(symbol, entry_price)
        except Exception as e:
            self._notify(f"❌ Sizing error for {symbol}: {e}")
            return
        
        if qty <= 0:
            print(f"[TM] {symbol}: zero quantity calculated, skipping")
            return
        
        # Calculate SL / TP prices
        sl_price = self._calc_sl_price(side, entry_price) if s.get('use_sl') else None
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
            return
        
        if not result:
            self._notify(f"❌ Order rejected for {symbol} {side}")
            return
        
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
        
        with self._lock:
            self._positions[symbol] = position
            # Init evaluator state alongside the position
            self._pos_state[symbol] = self._fresh_pos_state()
        self._persist_positions()
        
        self._notify_open(position)
    
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
            'opened_by': pos.get('opened_by', ''),
            'partial_closes_done': pos.get('partial_closes_done', []),
            # Carry the entry-side advisory snapshot into the closed record
            # so we can correlate "what the bot thought at open" with
            # "how the trade actually went". Critical for tuning weights
            # and validating whether Entry Score has predictive value.
            'entry_score': pos.get('entry_score'),
        }
        
        with self._lock:
            self._positions.pop(symbol, None)
            self._pos_state.pop(symbol, None)
            self._closed_trades.append(closed)
            if len(self._closed_trades) > CLOSED_TRADES_LIMIT:
                self._closed_trades = self._closed_trades[-CLOSED_TRADES_LIMIT:]
        self._persist_positions()
        self._persist_closed_trades()
        
        self._notify_close(closed)
    
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
    
    def _open_shadow(self, symbol: str, side: str, entry_price: float, opened_by: str):
        """Open a paper-trading position. No Bybit calls."""
        # Compute unified Decision Center verdict — same shape as real open
        decision = None
        try:
            decision = self.compute_decision(symbol, entry_price)
        except Exception as e:
            print(f"[TM] decision compute error on shadow open: {e}")
        opened_by_full = self._format_opened_by(opened_by, symbol,
                                                 entry_score=decision)
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': float(entry_price),
            'opened_at': time.time(),
            'opened_by': opened_by_full,
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
        msg = (
            f"{icon} <b>[TEST] OPEN {side}</b>: #{symbol}\n"
            f"📍 Entry: {self._fmt_price(entry_price)}\n"
            f"📋 {opened_by_full}\n"
            f"{es_block}"
            f"{ob_line}"
            f"🧪 Paper trading (no real order)"
        )
        self._notify(msg, is_test=True)
        print(f"[TM] [TEST] Shadow open: {symbol} {side} @ {self._fmt_price(entry_price)}")
    
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
        
        if filter_on:
            # Filter mode: short confirmation line — the gate already
            # validated direction match, so we just affirm it for the user.
            return f"🎯 OB Confirm: {bias_icon} {bias_label} ({tf_label}){age_str}\n"
        else:
            # Info-only mode: also show the zone bounds for context
            zone = (f"{self._fmt_price(row['bar_low'])}—"
                    f"{self._fmt_price(row['bar_high'])}")
            return f"🎯 OB: {bias_icon} {bias_label} ({tf_label}) zone {zone}{age_str}\n"
    
    def _format_entry_score_telegram(self, entry_score: Optional[Dict]) -> str:
        """Single-line Decision Center summary for Telegram OPEN messages.
        
        Was previously a multi-line breakdown — replaced with a compact
        one-liner per user feedback that the verbose form was confusing.
        Produces output like:
        
            🧠 Decision: LONG 78% (good)
            
        Empty string when entry_score is missing/disabled. Despite the
        legacy method name, this now reads from the Decision Center
        verdict shape (which extends entry_score with headline/verdict).
        """
        if not entry_score:
            return ''
        # If entry_score already has a 'headline' (Decision Center shape),
        # use it directly. Otherwise fall back to building one from raw
        # score/verdict (legacy entry_score shape).
        headline = entry_score.get('headline')
        verdict = entry_score.get('verdict', '?')
        if headline:
            return f"🧠 Decision: {headline} ({verdict})\n"
        score = entry_score.get('score', 0)
        sign = '+' if score >= 0 else ''
        return f"🧠 Decision: {sign}{score:.0f} ({verdict})\n"
    
    def _format_entry_score_recap(self, entry_score: Optional[Dict]) -> str:
        """Compact one-line recap of the Decision Center verdict for
        Telegram CLOSE messages. Format:
        
            🧠 Was: LONG 78% (good)
        """
        if not entry_score:
            return ''
        headline = entry_score.get('headline')
        verdict = entry_score.get('verdict', '?')
        if headline:
            return f"🧠 Was: {headline} ({verdict})\n"
        score = entry_score.get('score', 0)
        sign = '+' if score >= 0 else ''
        return f"🧠 Was: {sign}{score:.0f} ({verdict})\n"
    
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
        
        closed = {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': entry,
            'exit_price': float(exit_price),
            'opened_at': pos['opened_at'],
            'closed_at': time.time(),
            'pnl_pct': round(pnl_pct, 4),
            'reason': reason,
            'opened_by': pos.get('opened_by', ''),
            'shadow': True,
            # Same as real-position close — preserve the entry snapshot for
            # post-mortem analysis (was the entry score predictive?)
            'entry_score': pos.get('entry_score'),
        }
        
        with self._lock:
            self._shadow_positions.pop(symbol, None)
            self._shadow_pos_state.pop(symbol, None)
            self._shadow_closed.append(closed)
            if len(self._shadow_closed) > CLOSED_TRADES_LIMIT:
                self._shadow_closed = self._shadow_closed[-CLOSED_TRADES_LIMIT:]
        self._persist_shadow_positions()
        self._persist_shadow_closed()
        
        is_win = pnl_pct > 0
        icon = '✅' if is_win else '❌'
        # Recap the entry score so the user sees how the bot's pre-trade
        # assessment correlates with the actual outcome. This is what makes
        # post-mortem analysis useful — "I closed at -2%, but Entry Score was
        # 'good +50' — should I trust this score or down-weight it?"
        es_recap = self._format_entry_score_recap(pos.get('entry_score'))
        msg = (
            f"{icon} <b>[TEST] CLOSE {pos['side']}</b>: #{symbol}\n"
            f"📍 Entry: {self._fmt_price(entry)}\n"
            f"📤 Exit: {self._fmt_price(exit_price)}\n"
            f"💰 PnL: {pnl_pct:+.2f}%\n"
            f"🔖 Reason: {self._reason_label(reason)}\n"
            f"{es_recap}"
            f"🧪 Paper trade (no real close)"
        )
        self._notify(msg, is_test=True)
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
        
        # Round to reasonable precision (Bybit will reject too-fine qty;
        # an instrument-info lookup could be added later)
        if qty < 1:
            qty = round(qty, 6)
        else:
            qty = round(qty, 3)
        return max(0, qty)
    
    def _calc_sl_price(self, side: str, entry: float) -> float:
        sl_pct = self._settings.get('sl_pct', 2.0) / 100
        if side == 'LONG':
            return entry * (1 - sl_pct)
        return entry * (1 + sl_pct)
    
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
    
    # ============================================================
    # State / queries
    # ============================================================
    
    def get_state(self) -> Dict:
        with self._lock:
            positions = []
            for sym, pos in self._positions.items():
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
                # Attach Health Score (None when feature disabled)
                health = self._compute_health(pos, is_shadow=False)
                if health is not None:
                    pos_dict['health'] = health
                positions.append(pos_dict)
            
            closed = list(self._closed_trades[-50:])
            
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
                shadow_positions.append(pos_dict)
            shadow_closed = list(self._shadow_closed[-50:])
            
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
    
    def manual_close(self, symbol: str) -> Dict:
        """Manually close a position via UI button."""
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return {'ok': False, 'reason': 'No open position for symbol'}
        current = self._get_current_price(symbol) or pos['entry_price']
        self._close_position(symbol, current, reason='manual')
        return {'ok': True}
    
    def manual_close_shadow(self, symbol: str) -> Dict:
        """Manually close a paper-trading position via UI button."""
        with self._lock:
            pos = self._shadow_positions.get(symbol)
        if not pos:
            return {'ok': False, 'reason': 'No open paper position for symbol'}
        current = self._get_current_price(symbol) or pos['entry_price']
        self._close_shadow(symbol, current, reason='manual')
        return {'ok': True}
    
    def delete_closed_trade(self, idx: int) -> Dict:
        """Permanently remove a closed real trade by index. Stats are
        recomputed on the fly inside get_state(), so removing the entry from
        the list is enough — the next state poll will show updated PnL/win
        rate as if the trade never happened.
        """
        with self._lock:
            try:
                idx = int(idx)
            except Exception:
                return {'ok': False, 'reason': 'invalid index'}
            if idx < 0 or idx >= len(self._closed_trades):
                return {'ok': False, 'reason': 'index out of range'}
            removed = self._closed_trades.pop(idx)
        self._persist_closed_trades()
        return {'ok': True, 'removed': removed.get('symbol', '')}
    
    def delete_shadow_closed_trade(self, idx: int) -> Dict:
        """Permanently remove a closed paper trade by index."""
        with self._lock:
            try:
                idx = int(idx)
            except Exception:
                return {'ok': False, 'reason': 'invalid index'}
            if idx < 0 or idx >= len(self._shadow_closed):
                return {'ok': False, 'reason': 'index out of range'}
            removed = self._shadow_closed.pop(idx)
        self._persist_shadow_closed()
        return {'ok': True, 'removed': removed.get('symbol', '')}
    
    # ============================================================
    # Notifications
    # ============================================================
    
    def _notify(self, msg: str, is_test: bool = False):
        """Send Telegram notification, respecting the relevant toggle.
        
        Args:
            msg: text to send
            is_test: True if from shadow/paper trade — gated by test_telegram_alerts.
                     False if from real position — gated by telegram_alerts.
        """
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
        try:
            self.notifier.send_message(msg)
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
        opened_by_line = ''
        if pos.get('opened_by'):
            opened_by_line = f"📋 {pos['opened_by']}\n"
        msg = (
            f"{icon} <b>OPEN {side}</b>: #{pos['symbol']}\n"
            f"📍 Entry: {self._fmt_price(pos['entry_price'])}\n"
            f"💼 Qty: {pos['qty']:.6g}\n"
            f"🛡 SL: {sl_str}\n"
            f"🎯 TP: {tp_str}\n"
            f"⚙️ Lev: {self._settings.get('leverage', 10)}x\n"
            f"{opened_by_line}"
            f"{es_block}"
        ).rstrip() + '\n'
        self._notify(msg)
    
    def _notify_close(self, closed):
        side = closed['side']
        pnl_pct = closed['pnl_pct']
        pnl_usd = closed['pnl_usd']
        is_win = pnl_pct > 0
        icon = '✅' if is_win else '❌'
        # Show what the entry score said when we opened — useful for
        # weight-tuning and spotting when the predictor was wrong.
        es_recap = self._format_entry_score_recap(closed.get('entry_score'))
        msg = (
            f"{icon} <b>CLOSE {side}</b>: #{closed['symbol']}\n"
            f"📍 Entry: {self._fmt_price(closed['entry_price'])}\n"
            f"📤 Exit: {self._fmt_price(closed['exit_price'])}\n"
            f"💰 PnL: {pnl_pct:+.2f}% ({pnl_usd:+.2f}$)\n"
            f"🔖 Reason: {self._reason_label(closed['reason'])}\n"
            f"{es_recap}"
        ).rstrip()
        self._notify(msg)
    
    @staticmethod
    def _reason_label(reason: str) -> str:
        return {
            'stop_loss': '🛡 Stop Loss',
            'take_profit': '🎯 Take Profit',
            'reverse_smc': '🔄 Reverse SMC (CHoCH)',
            'reverse_signal': '🔁 Reverse Signal (qualified)',
            'forecast_1h_confluence': '🔮 Forecast 1H Confluence',
            'htf_flip': '📡 HTF Trend Flip',
            'time_stop': '⏱ Time Stop',
            'trailing_stop': '📈 Trailing Stop',
            'manual': '✋ Manual',
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
