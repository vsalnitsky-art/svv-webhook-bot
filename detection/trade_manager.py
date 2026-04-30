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
                      'health_score_enabled']:
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
        if not s.get('use_trailing'):
            return
        
        entry = pos['entry_price']
        side = pos['side']
        
        # Activation
        if not pos.get('trailing_active'):
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
        
        Behavior:
          - TM enabled + no existing real position → open real on Bybit
          - TM disabled + test_mode + no shadow position → open shadow (paper)
          - Reverse SMC handled in on_choch_event (event-level, not signal-level)
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
                # Already have a real position — Reverse handled by on_choch_event
                return
            if not self._is_tradeable(symbol):
                print(f"[TM] {symbol} not in tradeable list — signal ignored")
                return
            self._open_position(symbol, side, entry_price, opened_by)
        elif test_mode:
            # Paper mode — track shadow position
            if existing_shadow:
                # Already shadowing this symbol
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
    
    def _format_opened_by(self, opened_by: str, symbol: str) -> str:
        """Build a context-rich opened_by label that captures the state of
        Forecast 1H and CTR at the moment a position was opened. The base
        opened_by tag (e.g. 'choch_bos', 'choch') is preserved as the prefix
        so existing logic that inspects it keeps working — see _close_position
        and the reasonLabel mapping on the frontend.
        
        Format: '<base> · 🔮 <SIDE> +<pct>%·<conf>% · ⚡ <SIDE> <zone> <stc>'
        Missing data is rendered with em-dashes so the slot is always present.
        Examples:
            choch_bos · 🔮 LONG +25%·50% · ⚡ LONG Overbought 89
            choch · F:— · CTR:—
        """
        # Forecast 1H part
        fc = self._get_forecast_1h(symbol)
        if fc and fc.get('confidence', 0) > 0:
            side_n = fc.get('side', 0)
            side_lbl = 'LONG' if side_n > 0 else ('SHORT' if side_n < 0 else 'NEUTRAL')
            pct = int(fc.get('pct', 0) or 0)
            conf = int(fc.get('confidence', 0) or 0)
            sign = '+' if pct > 0 else ''
            fc_part = f"🔮 {side_lbl} {sign}{pct}%·{conf}%"
        else:
            fc_part = "F:—"
        
        # CTR part
        ctr = self._get_ctr(symbol)
        if ctr and ctr.get('stc') is not None:
            stc = ctr.get('stc')
            last_dir = ctr.get('last_dir') or '—'
            try:
                stc_int = int(round(float(stc)))
            except Exception:
                stc_int = stc
            if stc_int != '—' and isinstance(stc_int, (int, float)):
                if stc_int >= 75:
                    zone = 'Overbought'
                elif stc_int <= 25:
                    zone = 'Oversold'
                else:
                    zone = 'Mid'
                ctr_part = f"⚡ {last_dir} {zone} {stc_int}"
            else:
                ctr_part = "CTR:—"
        else:
            ctr_part = "CTR:—"
        
        return f"{opened_by} · {fc_part} · {ctr_part}"
    
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
        """Called by SMC scanner when a BOS event is detected (any).
        
        Two responsibilities:
          1) Update Health Score evaluator state (BOS counts in both directions)
             for real AND shadow positions, regardless of TM enabled state.
          2) Trigger BOS-N partial closes — only for real positions when TM is
             enabled and use_bos_partials is on.
        """
        # === Evaluator-state updates (always run if any position exists) ===
        s = self._settings
        with self._lock:
            real = self._positions.get(symbol)
            shadow = self._shadow_positions.get(symbol)
            
            for pos, state_dict in (
                (real, self._pos_state),
                (shadow, self._shadow_pos_state),
            ):
                if not pos or symbol not in state_dict:
                    continue
                pos_dir = 'bull' if pos['side'] == 'LONG' else 'bear'
                if direction == pos_dir:
                    state_dict[symbol]['bos_with_count'] = (
                        state_dict[symbol].get('bos_with_count', 0) + 1)
                else:
                    state_dict[symbol]['bos_against_count'] = (
                        state_dict[symbol].get('bos_against_count', 0) + 1)
        
        # === Partial-close logic (real positions only, requires TM enabled) ===
        if not self.is_enabled():
            return
        if not s.get('use_bos_partials'):
            return
        if not real:
            return
        
        # Only count BOS in same direction as our position for the partial logic
        bos_side = 'LONG' if direction == 'bull' else 'SHORT'
        if bos_side != real['side']:
            return
        
        # Increment count
        real['bos_count_since_entry'] = real.get('bos_count_since_entry', 1) + 1
        n = real['bos_count_since_entry']
        
        # Determine if this BOS triggers a partial
        partial_pct = s.get(f'bos_{n}_close_pct', 0)
        marker_key = f'bos_{n}'
        
        if partial_pct > 0 and marker_key not in real.get('partial_closes_done', []):
            current_price = self._get_current_price(symbol) or level
            self._partial_close(symbol, partial_pct, current_price,
                                reason=f'bos_{n}_partial')
            real.setdefault('partial_closes_done', []).append(marker_key)
            
            # Auto-activate trailing after BOS-2
            if n == 2 and s.get('trailing_after_bos_2'):
                real['trailing_active'] = True
                real['trailing_peak'] = current_price
                self._notify(f"📈 Trailing активовано після BOS-2: {symbol}")
            
            self._persist_positions()
    
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
        # Enrich opened_by with a snapshot of Forecast 1H + CTR state at open
        opened_by_full = self._format_opened_by(opened_by, symbol)
        position = _new_position(symbol, side, entry_price, qty,
                                   sl_price, tp_price, order_id, opened_by_full)
        
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
    
    # ============================================================
    # Shadow (paper) positions — for test_mode
    # ============================================================
    
    def _open_shadow(self, symbol: str, side: str, entry_price: float, opened_by: str):
        """Open a paper-trading position. No Bybit calls."""
        opened_by_full = self._format_opened_by(opened_by, symbol)
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': float(entry_price),
            'opened_at': time.time(),
            'opened_by': opened_by_full,
            'shadow': True,
        }
        with self._lock:
            self._shadow_positions[symbol] = pos
            self._shadow_pos_state[symbol] = self._fresh_pos_state()
        self._persist_shadow_positions()
        
        # Use colored circle for direction (instead of 📊)
        icon = '🟢' if side == 'LONG' else '🔴'
        msg = (
            f"{icon} <b>[TEST] OPEN {side}</b>: #{symbol}\n"
            f"📍 Entry: {self._fmt_price(entry_price)}\n"
            f"📋 {opened_by_full}\n"
            f"🧪 Paper trading (no real order)"
        )
        self._notify(msg, is_test=True)
        print(f"[TM] [TEST] Shadow open: {symbol} {side} @ {self._fmt_price(entry_price)}")
    
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
        msg = (
            f"{icon} <b>[TEST] CLOSE {pos['side']}</b>: #{symbol}\n"
            f"📍 Entry: {self._fmt_price(entry)}\n"
            f"📤 Exit: {self._fmt_price(exit_price)}\n"
            f"💰 PnL: {pnl_pct:+.2f}%\n"
            f"🔖 Reason: {self._reason_label(reason)}\n"
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
                },
                'shadow_positions': shadow_positions,
                'shadow_closed': shadow_closed,
                'shadow_stats': {
                    'total_closed': sh_total,
                    'wins': sh_wins,
                    'losses': sh_losses,
                    'win_rate': round(sh_wins / sh_total * 100, 1) if sh_total else 0,
                    'avg_pnl_pct': round(sh_avg_pnl, 2),
                },
                'settings': dict(self._settings),
            }
    
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
        msg = (
            f"{icon} <b>OPEN {side}</b>: #{pos['symbol']}\n"
            f"📍 Entry: {self._fmt_price(pos['entry_price'])}\n"
            f"💼 Qty: {pos['qty']:.6g}\n"
            f"🛡 SL: {sl_str}\n"
            f"🎯 TP: {tp_str}\n"
            f"⚙️ Lev: {self._settings.get('leverage', 10)}x"
        )
        self._notify(msg)
    
    def _notify_close(self, closed):
        side = closed['side']
        pnl_pct = closed['pnl_pct']
        pnl_usd = closed['pnl_usd']
        is_win = pnl_pct > 0
        icon = '✅' if is_win else '❌'
        msg = (
            f"{icon} <b>CLOSE {side}</b>: #{closed['symbol']}\n"
            f"📍 Entry: {self._fmt_price(closed['entry_price'])}\n"
            f"📤 Exit: {self._fmt_price(closed['exit_price'])}\n"
            f"💰 PnL: {pnl_pct:+.2f}% ({pnl_usd:+.2f}$)\n"
            f"🔖 Reason: {self._reason_label(closed['reason'])}"
        )
        self._notify(msg)
    
    @staticmethod
    def _reason_label(reason: str) -> str:
        return {
            'stop_loss': '🛡 Stop Loss',
            'take_profit': '🎯 Take Profit',
            'reverse_smc': '🔄 Reverse SMC (CHoCH)',
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
