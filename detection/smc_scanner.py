"""
Smart Money Scanner v1.1 — Background watchlist scanner with Telegram alerts.

For each symbol in the user's watchlist:
  1. Fetches 15m klines (300 bars) every 60 seconds
  2. Runs SMC structure detection TWICE:
     - On all bars (including the live unclosed one) → cached for chart display
     - On closed bars only (live bar dropped) → used for alert detection
  3. Compares against last detected event ID to find newly formed events
  4. Sends Telegram alert based on user's chosen alert mode

Why two passes?
  Pine `alertcondition()` defaults to fire once per bar close. A BOS/CHoCH that
  forms intra-bar but later "un-forms" before close would otherwise produce
  a false alert. Detecting on closed bars only prevents this. The chart still
  shows live structure for visual feedback.

Alert modes:
  'choch'       — Alert on every new CHoCH
  'choch_bos'   — Alert only when CHoCH is followed by a BOS in the same direction

Settings persisted in DB:
  smc_watchlist:    list of symbols
  smc_settings:     {alert_mode, interval_secs, enabled, telegram_alerts}

Uses MarketData (Binance → OKX → Bybit fallback) for klines.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional


# Defaults
DEFAULT_INTERVAL_SECS = 60
DEFAULT_ALERT_MODE = 'choch'  # 'choch' or 'choch_bos'
KLINES_LIMIT = 700            # bars to fetch per scan
DEFAULT_TIMEFRAME = '5m'      # default timeframe
DEFAULT_INTERNAL_SIZE = 5     # default Pine Internal Structure size
MAX_WATCHLIST = 50

# Allowed timeframes (Binance format)
ALLOWED_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '4h']
ALLOWED_HTF_METHODS = ['EMA Cross', 'EMA Trend', 'Swing Structure', 'Internal Structure']

# Display labels (uppercase) for UI / Telegram
TIMEFRAME_LABELS = {
    '1m': '1M', '3m': '3M', '5m': '5M', '15m': '15M',
    '30m': '30M', '1h': '1H', '4h': '4H',
}

DB_KEY_WATCHLIST = 'smc_watchlist'
DB_KEY_TRADEABLE = 'smc_tradeable'   # subset of watchlist that's tradeable
DB_KEY_SETTINGS = 'smc_settings'
DB_KEY_STATE = 'smc_last_events'   # tracks last seen event per symbol
DB_KEY_SIGNALS_PREFIX = 'smc_signals_'   # per-symbol: smc_signals_BTCUSDT, etc.
SIGNALS_PERSIST_LIMIT = 50         # max signals stored per symbol

DEFAULT_SETTINGS = {
    'enabled': True,
    'alert_mode': DEFAULT_ALERT_MODE,
    'interval_secs': DEFAULT_INTERVAL_SECS,
    'telegram_alerts': True,
    'swing_size': 50,
    'timeframe': DEFAULT_TIMEFRAME,
    'internal_size': DEFAULT_INTERNAL_SIZE,
    
    # Signal deduplication (Pine "Deduplicate Signals (1 per trend)").
    # When ON: only the FIRST signal in each direction is sent. Subsequent
    # same-direction signals are suppressed until direction flips.
    # User-requested default: True.
    'deduplicate_signals': True,
    
    # HTF Bias filter (Pine PRO HTF Bias group + Internal Structure addition)
    'htf_enabled': False,
    'htf_timeframe': '15m',
    'htf_method': 'EMA Trend',
    'htf_ema_fast': 9,
    'htf_ema_slow': 21,
    'htf_ema_trend': 50,
    # 'Internal Structure' method — runs SMC structure detection on HTF and
    # uses the LAST CHoCH event's direction as the trend. Pine "Deduplicate
    # 1 per trend, CHoCH only" semantics: BOS events don't change trend.
    'htf_internal_size': 3,
}


def _ema(values, period):
    """Pine ta.ema(): exponential moving average. Returns array of EMA values
    aligned with input. Initial value is the first close."""
    if not values or period < 1:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(alpha * values[i] + (1 - alpha) * out[-1])
    return out


def calc_htf_bias(htf_klines, method, ema_fast=9, ema_slow=21, ema_trend=50,
                   swing_trend=None, internal_size=3):
    """Compute HTF bias on the given HTF klines.
    
    Args:
        htf_klines: list of {p (close), ...} dicts at the HTF timeframe.
        method: 'EMA Cross' | 'EMA Trend' | 'Swing Structure' | 'Internal Structure'
        swing_trend: int (1=BULL, -1=BEAR, 0=NEUTRAL) for Swing Structure method
        internal_size: pivot size for Internal Structure method
    
    Returns:
        dict with:
            'bias': 'bull' | 'bear' | 'neutral'
            'method': method used
            'fast_value', 'slow_value', 'trend_value', 'close' (for debug)
            'last_choch_t' (for Internal Structure)
    """
    if method == 'Swing Structure':
        if swing_trend == 1:
            return {'bias': 'bull', 'method': method}
        elif swing_trend == -1:
            return {'bias': 'bear', 'method': method}
        else:
            return {'bias': 'neutral', 'method': method}
    
    if not htf_klines or len(htf_klines) < 2:
        return {'bias': 'neutral', 'method': method, 'reason': 'not enough klines'}
    
    # === Internal Structure method ===
    # Run SMC structure detection on the HTF klines (incl. live bar — real-time)
    # and look at the LAST CHoCH event. Its direction = HTF trend.
    # BOS events don't change the trend (they continue it).
    if method == 'Internal Structure':
        if len(htf_klines) < internal_size + 5:
            return {'bias': 'neutral', 'method': method, 'reason': 'not enough HTF bars'}
        try:
            from detection.smc_structure import detect_smc_structure
            result = detect_smc_structure(htf_klines, internal_size=internal_size,
                                            swing_size=50)
            events = result.get('internal', {}).get('events', [])
            # Find the most recent CHoCH (BOS ignored — only CHoCH changes trend)
            last_choch = None
            for e in reversed(events):
                if e.get('tag') == 'CHoCH':
                    last_choch = e
                    break
            if last_choch is None:
                return {'bias': 'neutral', 'method': method, 'reason': 'no CHoCH yet on HTF'}
            return {
                'bias': last_choch['dir'],   # 'bull' or 'bear'
                'method': method,
                'last_choch_t': last_choch.get('to_t'),
                'last_choch_level': last_choch.get('level'),
            }
        except Exception as e:
            return {'bias': 'neutral', 'method': method, 'reason': f'error: {e}'}
    
    closes = [k.get('p', 0) for k in htf_klines]
    last_close = closes[-1]
    
    if method == 'EMA Cross':
        if len(closes) < ema_slow:
            return {'bias': 'neutral', 'method': method, 'reason': 'not enough bars for slow EMA'}
        fast = _ema(closes, ema_fast)
        slow = _ema(closes, ema_slow)
        is_bull = fast[-1] > slow[-1]
        return {
            'bias': 'bull' if is_bull else 'bear',
            'method': method,
            'fast_value': round(fast[-1], 6),
            'slow_value': round(slow[-1], 6),
            'close': round(last_close, 6),
        }
    
    # 'EMA Trend' (default)
    if len(closes) < ema_trend:
        return {'bias': 'neutral', 'method': method, 'reason': 'not enough bars for trend EMA'}
    trend = _ema(closes, ema_trend)
    is_bull = last_close > trend[-1]
    return {
        'bias': 'bull' if is_bull else 'bear',
        'method': method,
        'trend_value': round(trend[-1], 6),
        'close': round(last_close, 6),
    }


class SMCScanner:
    
    def __init__(self, db=None, notifier=None):
        self.db = db
        self.notifier = notifier
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Cache of latest analysis per symbol (for instant chart display)
        self._cache: Dict[str, Dict] = {}
        
        # Pending CHoCH events waiting for BOS confirmation (mode 'choch_bos')
        # {symbol: {'from_t', 'level', 'dir', 'tag'}}
        self._pending_choch: Dict[str, Dict] = {}
        
        # Last alerted event identifier per (symbol, kind)
        # Identifier = (from_t, level, tag) tuple as string
        self._last_alerted: Dict[str, str] = {}
        
        # Track which symbols have completed their first scan.
        # On the first scan we only RECORD existing events as "already seen",
        # but do NOT send alerts. This prevents spam at startup or when a
        # symbol is added to the watchlist.
        # {symbol: set_of_event_ids_seen_on_first_scan}
        self._first_scan_done: Dict[str, bool] = {}
        self._seen_events: Dict[str, set] = {}  # {symbol: {ev_id, ...}}
        
        # HTF Bias cache per symbol: {symbol: {'bias', 'method', ...}}
        self._htf_cache: Dict[str, Dict] = {}
        
        # Signal markers per symbol — points on the chart where Telegram alerts
        # actually fired. Used to display LONG/SHORT dots on the chart.
        # {symbol: [{'time': ts_sec, 'price': float, 'side': 'LONG'|'SHORT'}, ...]}
        # Persisted in DB per-symbol so they survive restarts.
        self._signal_markers: Dict[str, List[Dict]] = {}
        
        # Last sent signal direction per symbol — used by deduplicate logic.
        # {symbol: 'LONG' | 'SHORT' | None}.
        # Seeded from the most recent persisted signal marker at startup, so
        # dedup state survives restarts.
        self._last_signal_dir: Dict[str, str] = {}
        
        self._scan_count = 0
        self._errors = 0
        
        self._settings = self._load_settings()
        self._watchlist = self._load_watchlist()
        self._tradeable = self._load_tradeable()
        # Load signals only for symbols currently in watchlist
        self._load_all_signals()
    
    # ========================================
    # Persistence
    # ========================================
    
    def _load_settings(self) -> Dict:
        if not self.db:
            return DEFAULT_SETTINGS.copy()
        try:
            stored = self.db.get_setting(DB_KEY_SETTINGS, None)
            if isinstance(stored, dict):
                merged = DEFAULT_SETTINGS.copy()
                merged.update(stored)
                return merged
        except:
            pass
        return DEFAULT_SETTINGS.copy()
    
    def _load_watchlist(self) -> List[str]:
        if not self.db:
            return []
        try:
            stored = self.db.get_setting(DB_KEY_WATCHLIST, [])
            if isinstance(stored, list):
                return [s for s in stored if isinstance(s, str)]
        except:
            pass
        return []
    
    def _persist_settings(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_SETTINGS, self._settings)
            except Exception as e:
                print(f"[SMC] Settings persist error: {e}")
    
    def _persist_watchlist(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_WATCHLIST, self._watchlist)
            except Exception as e:
                print(f"[SMC] Watchlist persist error: {e}")
    
    def _load_tradeable(self) -> List[str]:
        """List of symbols flagged as tradeable for Trade Manager."""
        if not self.db:
            return []
        try:
            stored = self.db.get_setting(DB_KEY_TRADEABLE, [])
            if isinstance(stored, list):
                return [s for s in stored if isinstance(s, str)]
        except:
            pass
        return []
    
    def _persist_tradeable(self):
        if self.db:
            try:
                self.db.set_setting(DB_KEY_TRADEABLE, self._tradeable)
            except Exception as e:
                print(f"[SMC] Tradeable persist error: {e}")
    
    def _load_all_signals(self):
        """Load persisted signal markers for every watchlist symbol."""
        if not self.db:
            return
        loaded = 0
        for symbol in self._watchlist:
            try:
                key = DB_KEY_SIGNALS_PREFIX + symbol
                stored = self.db.get_setting(key, [])
                if isinstance(stored, list):
                    # Validate items
                    cleaned = []
                    for item in stored:
                        if isinstance(item, dict) and 'time' in item and 'side' in item:
                            cleaned.append({
                                'time': int(item['time']),
                                'price': float(item.get('price', 0)),
                                'side': str(item['side']),
                            })
                    if cleaned:
                        self._signal_markers[symbol] = cleaned
                        loaded += len(cleaned)
                        # Seed last_signal_dir from the most recent marker
                        # so dedup state survives restarts
                        try:
                            last = max(cleaned, key=lambda m: m['time'])
                            self._last_signal_dir[symbol] = last['side']
                        except:
                            pass
            except Exception as e:
                print(f"[SMC] Signal load error for {symbol}: {e}")
        if loaded:
            print(f"[SMC] Loaded {loaded} persisted signal markers across "
                  f"{len(self._signal_markers)} symbols")
    
    def _persist_signals(self, symbol: str):
        """Save signal markers for one symbol to DB."""
        if not self.db:
            return
        try:
            key = DB_KEY_SIGNALS_PREFIX + symbol
            markers = self._signal_markers.get(symbol, [])
            self.db.set_setting(key, markers)
        except Exception as e:
            print(f"[SMC] Signal persist error for {symbol}: {e}")
    
    def clear_signals(self, symbol: Optional[str] = None) -> Dict:
        """Clear persisted signal markers. If symbol is None, clear ALL.
        Also resets dedup tracking so the next signal in any direction fires.
        """
        with self._lock:
            if symbol:
                sym = self._normalize_symbol(symbol)
                self._signal_markers.pop(sym, None)
                self._last_signal_dir.pop(sym, None)
                self._delete_signals(sym)
                return {'ok': True, 'cleared': sym}
            else:
                cleared = list(self._signal_markers.keys())
                self._signal_markers.clear()
                self._last_signal_dir.clear()
                for s in cleared:
                    self._delete_signals(s)
                return {'ok': True, 'cleared': cleared, 'count': len(cleared)}
    
    def _delete_signals(self, symbol: str):
        """Remove persisted signals for a symbol (e.g. on watchlist remove)."""
        if not self.db:
            return
        try:
            key = DB_KEY_SIGNALS_PREFIX + symbol
            # Set to empty list — DB doesn't have a delete API in our codebase
            self.db.set_setting(key, [])
        except Exception as e:
            print(f"[SMC] Signal delete error for {symbol}: {e}")
    
    # ========================================
    # Public API — Watchlist
    # ========================================
    
    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self._watchlist)
    
    def get_tradeable_symbols(self) -> List[str]:
        with self._lock:
            # Only return tradeable that are also in watchlist (avoid stale)
            return [s for s in self._tradeable if s in self._watchlist]
    
    def set_tradeable(self, symbol: str, tradeable: bool) -> Dict:
        symbol = self._normalize_symbol(symbol)
        with self._lock:
            if symbol not in self._watchlist:
                return {'ok': False, 'reason': 'Symbol not in watchlist'}
            
            currently = symbol in self._tradeable
            if tradeable and not currently:
                self._tradeable.append(symbol)
                self._persist_tradeable()
            elif not tradeable and currently:
                self._tradeable.remove(symbol)
                self._persist_tradeable()
            
            return {'ok': True, 'symbol': symbol,
                    'tradeable': symbol in self._tradeable}
    
    def add_symbol(self, symbol: str) -> Dict:
        symbol = self._normalize_symbol(symbol)
        if not symbol:
            return {'ok': False, 'reason': 'Invalid symbol'}
        
        with self._lock:
            if symbol in self._watchlist:
                return {'ok': False, 'reason': 'Already in watchlist', 'symbol': symbol}
            if len(self._watchlist) >= MAX_WATCHLIST:
                return {'ok': False, 'reason': f'Max {MAX_WATCHLIST} symbols'}
            
            # Validate via Binance
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                klines = md.fetch_klines(symbol, limit=10)
                if not klines:
                    return {'ok': False, 'reason': 'Symbol not found on any exchange'}
            except Exception as e:
                return {'ok': False, 'reason': f'Validation error: {e}'}
            
            self._watchlist.append(symbol)
            self._persist_watchlist()
            return {'ok': True, 'symbol': symbol, 'watchlist': list(self._watchlist)}
    
    def remove_symbol(self, symbol: str) -> Dict:
        symbol = self._normalize_symbol(symbol)
        with self._lock:
            if symbol in self._watchlist:
                self._watchlist.remove(symbol)
                self._persist_watchlist()
                # Clean up state
                self._pending_choch.pop(symbol, None)
                self._first_scan_done.pop(symbol, None)
                self._seen_events.pop(symbol, None)
                self._htf_cache.pop(symbol, None)
                self._signal_markers.pop(symbol, None)
                self._last_signal_dir.pop(symbol, None)
                self._cache.pop(symbol, None)
                # Tradeable cleanup
                if symbol in self._tradeable:
                    self._tradeable.remove(symbol)
                    self._persist_tradeable()
                for k in list(self._last_alerted.keys()):
                    if k.startswith(f"{symbol}:"):
                        del self._last_alerted[k]
                # Clear persisted signals from DB
                self._delete_signals(symbol)
                return {'ok': True, 'watchlist': list(self._watchlist)}
        return {'ok': False, 'reason': 'Not in watchlist'}
    
    def _normalize_symbol(self, s: str) -> str:
        if not s:
            return ''
        s = s.strip().upper().replace('.P', '').replace(' ', '')
        if not s:
            return ''
        if not s.endswith('USDT'):
            s += 'USDT'
        return s
    
    # ========================================
    # Public API — Settings
    # ========================================
    
    def get_settings(self) -> Dict:
        with self._lock:
            return dict(self._settings)
    
    def update_settings(self, new: Dict) -> Dict:
        with self._lock:
            allowed = ['enabled', 'alert_mode', 'interval_secs', 'telegram_alerts',
                       'swing_size', 'timeframe', 'internal_size',
                       'deduplicate_signals',
                       'htf_enabled', 'htf_timeframe', 'htf_method',
                       'htf_ema_fast', 'htf_ema_slow', 'htf_ema_trend',
                       'htf_internal_size']
            
            # Detect changes that require cache reset
            old_tf = self._settings.get('timeframe', DEFAULT_TIMEFRAME)
            old_isize = self._settings.get('internal_size', DEFAULT_INTERNAL_SIZE)
            old_htf_tf = self._settings.get('htf_timeframe', '15m')
            old_htf_method = self._settings.get('htf_method', 'EMA Trend')
            old_htf_isize = self._settings.get('htf_internal_size', 3)
            
            for k in allowed:
                if k in new:
                    self._settings[k] = new[k]
            
            # Validate alert_mode
            if self._settings.get('alert_mode') not in ('choch', 'choch_bos'):
                self._settings['alert_mode'] = DEFAULT_ALERT_MODE
            
            # Clamp interval
            try:
                self._settings['interval_secs'] = max(30, min(600, int(self._settings.get('interval_secs', 60))))
            except:
                self._settings['interval_secs'] = DEFAULT_INTERVAL_SECS
            
            # Clamp swing_size
            try:
                self._settings['swing_size'] = max(10, min(200, int(self._settings.get('swing_size', 50))))
            except:
                self._settings['swing_size'] = 50
            
            # Validate timeframe
            if self._settings.get('timeframe') not in ALLOWED_TIMEFRAMES:
                self._settings['timeframe'] = DEFAULT_TIMEFRAME
            
            # Clamp internal_size
            try:
                self._settings['internal_size'] = max(2, min(20, int(self._settings.get('internal_size', 5))))
            except:
                self._settings['internal_size'] = DEFAULT_INTERNAL_SIZE
            
            # === HTF settings validation ===
            # htf_enabled is bool — coerce
            self._settings['htf_enabled'] = bool(self._settings.get('htf_enabled', False))
            
            # deduplicate_signals — coerce to bool
            self._settings['deduplicate_signals'] = bool(self._settings.get('deduplicate_signals', True))
            
            # htf_timeframe — same allowed set + 1d
            allowed_htf_tfs = ALLOWED_TIMEFRAMES + ['1d']
            if self._settings.get('htf_timeframe') not in allowed_htf_tfs:
                self._settings['htf_timeframe'] = '15m'
            
            # htf_method
            if self._settings.get('htf_method') not in ALLOWED_HTF_METHODS:
                self._settings['htf_method'] = 'EMA Trend'
            
            # EMA periods
            try:
                self._settings['htf_ema_fast'] = max(3, min(100, int(self._settings.get('htf_ema_fast', 9))))
            except:
                self._settings['htf_ema_fast'] = 9
            try:
                self._settings['htf_ema_slow'] = max(5, min(200, int(self._settings.get('htf_ema_slow', 21))))
            except:
                self._settings['htf_ema_slow'] = 21
            try:
                self._settings['htf_ema_trend'] = max(10, min(200, int(self._settings.get('htf_ema_trend', 50))))
            except:
                self._settings['htf_ema_trend'] = 50
            
            # Internal Structure size (Pine SMC pivot size on HTF)
            try:
                self._settings['htf_internal_size'] = max(2, min(20, int(self._settings.get('htf_internal_size', 3))))
            except:
                self._settings['htf_internal_size'] = 3
            
            # Reset cache on relevant changes
            new_tf = self._settings['timeframe']
            new_isize = self._settings['internal_size']
            new_htf_tf = self._settings['htf_timeframe']
            new_htf_method = self._settings['htf_method']
            new_htf_isize = self._settings['htf_internal_size']
            
            tf_changed = (new_tf != old_tf or new_isize != old_isize)
            htf_changed = (new_htf_tf != old_htf_tf or 
                           new_htf_method != old_htf_method or
                           new_htf_isize != old_htf_isize)
            
            if tf_changed:
                self._cache.clear()
                self._first_scan_done.clear()
                self._seen_events.clear()
                self._pending_choch.clear()
                # Don't drop signal_markers from DB — they represent historical
                # facts (alerts that did fire). Just reload from DB to keep
                # in-memory copy fresh.
                self._signal_markers.clear()
                self._load_all_signals()
                print(f"[SMC] Settings changed: tf {old_tf}→{new_tf}, "
                      f"size {old_isize}→{new_isize}. Cache cleared.")
            if htf_changed or tf_changed:
                self._htf_cache.clear()
                if htf_changed:
                    print(f"[SMC] HTF changed: {old_htf_tf}/{old_htf_method} → "
                          f"{new_htf_tf}/{new_htf_method}. HTF cache cleared.")
            
            self._persist_settings()
            return dict(self._settings)
    
    def get_timeframe(self) -> str:
        return self._settings.get('timeframe', DEFAULT_TIMEFRAME)
    
    def get_display_label(self) -> str:
        return TIMEFRAME_LABELS.get(self.get_timeframe(), self.get_timeframe().upper())
    
    def get_internal_size(self) -> int:
        try:
            return int(self._settings.get('internal_size', DEFAULT_INTERNAL_SIZE))
        except:
            return DEFAULT_INTERNAL_SIZE
    
    def get_htf_settings(self) -> Dict:
        return {
            'enabled': bool(self._settings.get('htf_enabled', False)),
            'timeframe': self._settings.get('htf_timeframe', '15m'),
            'method': self._settings.get('htf_method', 'EMA Trend'),
            'ema_fast': int(self._settings.get('htf_ema_fast', 9)),
            'ema_slow': int(self._settings.get('htf_ema_slow', 21)),
            'ema_trend': int(self._settings.get('htf_ema_trend', 50)),
            'internal_size': int(self._settings.get('htf_internal_size', 3)),
        }
    
    def is_enabled(self) -> bool:
        return self._settings.get('enabled', True)
    
    def set_enabled(self, enabled: bool):
        self._settings['enabled'] = enabled
        self._persist_settings()
        if enabled and not self._running:
            self.start()
        elif not enabled and self._running:
            self._running = False
    
    # ========================================
    # Lifecycle
    # ========================================
    
    def start(self):
        if self._running:
            return
        if not self.is_enabled():
            print("[SMC] Disabled in settings, not starting")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SMCScanner")
        self._thread.start()
        print(f"[SMC] ✅ Started: timeframe={self.get_timeframe()}, "
              f"interval={self._settings['interval_secs']}s, "
              f"watchlist={len(self._watchlist)}, "
              f"alert_mode={self._settings['alert_mode']}")
    
    def stop(self):
        self._running = False
    
    def _loop(self):
        print("[SMC] 🧵 Thread started")
        time.sleep(15)  # initial delay
        while self._running:
            try:
                self._scan()
            except Exception as e:
                self._errors += 1
                if self._errors <= 5:
                    print(f"[SMC] Scan error: {e}")
            
            interval = self._settings.get('interval_secs', DEFAULT_INTERVAL_SECS)
            for _ in range(interval):
                if not self._running:
                    return
                time.sleep(1)
    
    def _scan(self):
        if not self._watchlist:
            return
        
        from detection.market_data import get_market_data
        from detection.smc_structure import detect_smc_structure
        
        md = get_market_data()
        self._scan_count += 1
        
        for symbol in list(self._watchlist):
            if not self._running:
                return
            try:
                # Fetch klines at configured timeframe
                tf = self.get_timeframe()
                klines = md.fetch_klines(symbol, limit=KLINES_LIMIT, interval=tf) \
                    if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames \
                    else md.fetch_klines(symbol, limit=KLINES_LIMIT)
                
                if not klines or len(klines) < 50:
                    continue
                
                # ┌─────────────────────────────────────────────────────────┐
                # │ Two analyses, two purposes — matches TradingView:       │
                # │                                                         │
                # │ result_full     — live structure (all bars incl. live   │
                # │                   unclosed bar). Used for CHART display │
                # │                   so users see the same thing they'd    │
                # │                   see on TradingView in real time.      │
                # │                                                         │
                # │ result_closed   — confirmed structure (last bar dropped │
                # │                   so it's "frozen"). Used for ALERTS    │
                # │                   only. Mirrors Pine's default          │
                # │                   alertcondition() behavior of firing   │
                # │                   only on bar close.                    │
                # │                                                         │
                # │ Why both? On the chart we want responsiveness — see     │
                # │ structure form in real time. For alerts we want         │
                # │ stability — no signal until the bar is actually closed. │
                # │ This is exactly how TV indicators behave.               │
                # └─────────────────────────────────────────────────────────┘
                isize = self.get_internal_size()
                ssize = int(self._settings.get('swing_size', 50))
                
                result_full = detect_smc_structure(klines, internal_size=isize,
                                                     swing_size=ssize)
                
                klines_closed = klines[:-1] if len(klines) > 1 else klines
                result_closed = detect_smc_structure(klines_closed, internal_size=isize,
                                                       swing_size=ssize)
                
                # Cache full klines + full structure for chart display
                with self._lock:
                    self._cache[symbol] = {
                        'klines': klines,
                        'analysis': result_full,
                        'updated_at': time.time(),
                    }
                
                # === Compute HTF Bias (used by alert filter) ===
                htf_settings = self.get_htf_settings()
                htf_bias = self._compute_htf_bias(symbol, md, htf_settings,
                                                    swing_trend=result_closed.get('swing', {}).get('trend', 0))
                with self._lock:
                    self._htf_cache[symbol] = htf_bias
                
                # === Update Forecast 1H + CTR via forecast engine ===
                # Best-effort, never blocks scan loop on error
                try:
                    from detection.forecast_engine import get_forecast_engine
                    fe = get_forecast_engine()
                    if fe:
                        fe.update(symbol, ltf_klines=klines)
                except Exception as fe_err:
                    if self._errors <= 5:
                        print(f"[SMC] Forecast update error for {symbol}: {fe_err}")
                
                # Alerts run on CLOSED bars only — won't fire from intra-bar
                # wicks that retract before close
                self._process_alerts(symbol, result_closed)
                
                # 200ms between symbols to spread load
                time.sleep(0.2)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[SMC] {symbol} scan error: {e}")
                self._errors += 1
        
        if self._scan_count <= 2 or self._scan_count % 30 == 0:
            print(f"[SMC] Scan #{self._scan_count}: {len(self._watchlist)} symbols, errors={self._errors}")
    
    def _compute_htf_bias(self, symbol: str, md, htf_settings: Dict,
                            swing_trend: int = 0) -> Dict:
        """Fetch HTF klines and compute bias. Returns the result dict.
        
        For 'Swing Structure' method: uses swing_trend from current chart's
        SMC analysis — no extra API call needed.
        For 'Internal Structure' method: fetches HTF klines (incl. live bar
        for real-time response) and runs SMC detection.
        """
        method = htf_settings['method']
        
        # Swing Structure method doesn't need HTF data
        if method == 'Swing Structure':
            return calc_htf_bias(None, method, swing_trend=swing_trend)
        
        # EMA / Internal Structure all need HTF klines
        try:
            htf_tf = htf_settings['timeframe']
            
            # Bar count budget per method
            if method == 'Internal Structure':
                # Need enough bars for stable structure: 200 is generous
                need_bars = 200
            else:
                # EMA methods: longest period + buffer
                need_bars = max(htf_settings['ema_slow'], htf_settings['ema_trend']) + 50
            need_bars = min(need_bars, 500)
            
            htf_klines = md.fetch_klines(symbol, limit=need_bars, interval=htf_tf) \
                if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames \
                else None
            
            if not htf_klines:
                return {'bias': 'neutral', 'method': method, 'reason': 'fetch failed'}
            
            return calc_htf_bias(
                htf_klines, method,
                ema_fast=htf_settings['ema_fast'],
                ema_slow=htf_settings['ema_slow'],
                ema_trend=htf_settings['ema_trend'],
                internal_size=htf_settings.get('internal_size', 3),
            )
        except Exception as e:
            return {'bias': 'neutral', 'method': method, 'reason': f'error: {e}'}
    
    # ========================================
    # Alerts logic
    # ========================================
    
    def _process_alerts(self, symbol: str, result: Dict):
        if not self._settings.get('telegram_alerts', True) or not self.notifier:
            return
        
        events = result.get('internal', {}).get('events', [])
        if not events:
            self._first_scan_done[symbol] = True
            return
        
        # === Stable event identifier ===
        # IMPORTANT: We use ONLY (from_t, dir) — NOT tag.
        # Reason: when the algorithm re-runs over a growing klines array,
        # the SAME pivot point may be re-classified between CHoCH and BOS
        # depending on what came before in the new window. If we included
        # tag in the key, the same logical event would appear "new" again
        # and trigger duplicate alerts (this caused the false NEOUSDT BOS
        # confirmation that wasn't visible on chart).
        def ev_id(e):
            return f"{e.get('from_t')}:{e.get('dir')}"
        
        seen = self._seen_events.setdefault(symbol, set())
        is_first_scan = not self._first_scan_done.get(symbol, False)
        
        # Determine truly NEW events (their stable IDs aren't in `seen`)
        new_events = []
        for ev in events:
            eid = ev_id(ev)
            if eid not in seen:
                new_events.append((eid, ev))
                seen.add(eid)
        
        # Bound seen set to avoid unbounded growth
        if len(seen) > 200:
            recent_ids = {ev_id(e) for e in events[-200:]}
            self._seen_events[symbol] = recent_ids
        
        if is_first_scan:
            # First scan: just record existing events. NEVER seed pending_choch
            # from history, because the BOS that "confirms" it would also be
            # historical and is recorded as seen on this same scan.
            self._first_scan_done[symbol] = True
            print(f"[SMC] {symbol}: first scan recorded {len(events)} historical events "
                  f"(no alerts, no pending CHoCH from history)")
            return
        
        if not new_events:
            return  # nothing changed since last scan
        
        # === Recency guard ===
        # Only alert on events whose `to_t` (bar where BOS/CHoCH was confirmed
        # by close cross) is within the last few closed bars.
        # 5m timeframe → 30 min window (6 closed bars) is plenty fresh,
        # filters out events that "appeared" from history due to algorithm
        # re-evaluation across scans.
        recent_threshold_secs = 30 * 60  # 30 minutes
        now_ms = int(time.time() * 1000)
        
        mode = self._settings.get('alert_mode', DEFAULT_ALERT_MODE)
        
        for _, ev in new_events:
            tag = ev.get('tag')
            to_t = ev.get('to_t', 0) or 0
            # to_t is in ms (kline open time)
            to_age_secs = (now_ms - to_t) / 1000 if to_t > 1e10 else (now_ms / 1000 - to_t)
            
            is_recent = to_age_secs <= recent_threshold_secs
            
            if not is_recent:
                # Event is old — silently record but don't alert
                # (this was the bug: we used to alert on these)
                if mode == 'choch_bos' and tag == 'CHoCH':
                    # Don't even seed pending_choch from old CHoCH —
                    # only fresh CHoCH should anchor a future BOS confirmation
                    pass
                continue
            
            # === Forward BOS events to Trade Manager (always, before mode logic) ===
            # TM uses these to count BOS-N for partial closes after position open.
            # Independent of alert mode and dedup — TM has its own logic.
            if tag == 'BOS':
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm and tm.is_enabled():
                        tm.on_bos_event(symbol=symbol, direction=ev['dir'],
                                          level=ev['level'], bar_t=to_t)
                except Exception as e:
                    print(f"[SMC] TM BOS hook error: {e}")
            
            # === Forward ALL CHoCH events to TM (regardless of TM enabled) ===
            # TM uses these for: Reverse SMC exit, Forecast 1H Confluence exit.
            # Both rules can run in shadow mode (TM disabled but test_mode on)
            # to send Telegram-only signals without opening real positions.
            if tag == 'CHoCH':
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm:
                        tm.on_choch_event(symbol=symbol, direction=ev['dir'],
                                           level=ev['level'], bar_t=to_t)
                except Exception as e:
                    print(f"[SMC] TM CHoCH hook error: {e}")
            
            if mode == 'choch':
                if tag == 'CHoCH':
                    if not self._htf_allows(symbol, ev['dir']):
                        print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by HTF filter")
                    elif not self._dedup_allows(symbol, ev['dir']):
                        print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by dedup (already fired this direction)")
                    else:
                        self._send_alert(symbol, ev, mode='choch')
            
            elif mode == 'choch_bos':
                if tag == 'CHoCH':
                    # Fresh CHoCH — anchor for future BOS confirmation
                    self._pending_choch[symbol] = {
                        'from_t': ev['from_t'],
                        'to_t': ev['to_t'],
                        'level': ev['level'],
                        'dir': ev['dir'],
                        'choch_event': ev,
                    }
                elif tag == 'BOS':
                    pending = self._pending_choch.get(symbol)
                    if pending and pending['dir'] == ev['dir']:
                        # Additional safety: BOS must be AFTER the CHoCH chronologically
                        if ev.get('to_t', 0) > pending.get('to_t', 0):
                            if not self._htf_allows(symbol, ev['dir']):
                                print(f"[SMC] {symbol} CHoCH+BOS {ev['dir']} blocked by HTF filter")
                                # Don't pop pending — let next BOS in same direction try again
                            elif not self._dedup_allows(symbol, ev['dir']):
                                print(f"[SMC] {symbol} CHoCH+BOS {ev['dir']} blocked by dedup")
                                self._pending_choch.pop(symbol, None)
                            else:
                                self._send_alert(symbol, ev, mode='choch_bos',
                                                  choch_event=pending['choch_event'])
                                self._pending_choch.pop(symbol, None)
                    elif pending and pending['dir'] != ev['dir']:
                        # Opposite-direction BOS invalidates pending CHoCH
                        self._pending_choch.pop(symbol, None)
    
    def _dedup_allows(self, symbol: str, event_dir: str) -> bool:
        """Check if signal deduplication permits an alert in this direction.
        
        Returns True if dedup is OFF, or if direction differs from last signal.
        Returns False only when dedup is ON AND the same direction was last
        signaled (Pine 'Deduplicate Signals (1 per trend)' behavior).
        """
        if not self._settings.get('deduplicate_signals', True):
            return True
        last_side = self._last_signal_dir.get(symbol)
        # event_dir is 'bull' / 'bear'; last_side is 'LONG' / 'SHORT'
        side_label = 'LONG' if event_dir == 'bull' else 'SHORT'
        if last_side is None:
            return True  # first signal ever for this symbol
        return last_side != side_label
    
    def _htf_allows(self, symbol: str, event_dir: str) -> bool:
        """Check if HTF bias permits an alert in the given direction.
        
        Returns True if filter is OFF, or if direction matches HTF bias.
        Returns False only when filter is ON and direction is opposite.
        Neutral HTF bias is treated as "allow" so symbols with insufficient
        data don't get blocked permanently.
        """
        htf = self.get_htf_settings()
        if not htf['enabled']:
            return True
        
        bias = self._htf_cache.get(symbol, {}).get('bias', 'neutral')
        if bias == 'neutral':
            return True  # don't block when HTF data is unavailable
        
        # event_dir = 'bull' | 'bear'; bias = 'bull' | 'bear'
        return bias == event_dir
    
    def _send_alert(self, symbol: str, event: Dict, mode: str, choch_event: Dict = None):
        try:
            is_bull = event['dir'] == 'bull'
            dir_icon = '🟢' if is_bull else '🔴'
            side_label = 'LONG' if is_bull else 'SHORT'
            
            # Live entry price — close of the bar where the BOS/CHoCH crossover
            # occurred. This is the price at the actual moment the signal fired.
            entry_price = self._get_live_price(symbol)
            if entry_price is None:
                entry_price = event.get('level', 0)
            
            entry_str = self._fmt_price(entry_price)
            
            msg = (
                f"{dir_icon} <b>{side_label}</b>: #{symbol}\n"
                f"📍 Вхід: {entry_str}"
            )
            
            self.notifier.send_message(msg)
            
            # Record signal marker for chart display
            # Use to_t (timestamp of bar where crossover happened) as the time
            to_t = event.get('to_t', 0)
            t_sec = int(to_t // 1000) if to_t > 1e12 else int(to_t)
            persisted = False
            with self._lock:
                markers = self._signal_markers.setdefault(symbol, [])
                # Dedup — don't add the same time+side twice
                if not any(m['time'] == t_sec and m['side'] == side_label for m in markers):
                    markers.append({
                        'time': t_sec,
                        'price': float(entry_price),
                        'side': side_label,
                    })
                    # Keep last SIGNALS_PERSIST_LIMIT
                    if len(markers) > SIGNALS_PERSIST_LIMIT:
                        self._signal_markers[symbol] = markers[-SIGNALS_PERSIST_LIMIT:]
                    persisted = True
            
            # Save to DB outside the lock to avoid holding it during I/O
            if persisted:
                self._persist_signals(symbol)
            
            # Update last-direction state (used by dedup gate). Pine updates
            # this even when dedup is OFF, so toggling dedup ON later doesn't
            # cause a sudden re-fire of the prior direction.
            self._last_signal_dir[symbol] = side_label
            
            print(f"[SMC] 📨 Alert sent: {symbol} {side_label} @ {entry_str}")
            
            # === Forward signal to Trade Manager ===
            try:
                from detection.trade_manager import get_trade_manager
                tm = get_trade_manager()
                if tm and tm.is_enabled():
                    tm.on_signal(symbol=symbol, side=side_label,
                                 entry_price=entry_price, opened_by=mode)
            except Exception as e:
                print(f"[SMC] TM hook error: {e}")
        except Exception as e:
            print(f"[SMC] Alert send error: {e}")
    
    def _get_live_price(self, symbol: str) -> Optional[float]:
        """Return the most recent close price for the symbol.
        
        Uses the cached klines if available (last bar's close = newest price
        at the time of the most recent scan). Returns None if no cache.
        """
        with self._lock:
            cached = self._cache.get(symbol)
        if not cached or not cached.get('klines'):
            return None
        try:
            return float(cached['klines'][-1].get('p', 0))
        except:
            return None
    
    def _fmt_price(self, price: float) -> str:
        """Format price with appropriate precision."""
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
    
    # ========================================
    # Public API — Chart data
    # ========================================
    
    def get_chart_data(self, symbol: str) -> Dict:
        """Return klines + structure for chart rendering. Uses cache if fresh."""
        symbol = self._normalize_symbol(symbol)
        
        with self._lock:
            cached = self._cache.get(symbol)
        
        # Use cache if < 30s old
        if cached and (time.time() - cached['updated_at']) < 30:
            return self._format_chart(symbol, cached['klines'], cached['analysis'])
        
        # Otherwise fetch fresh
        try:
            from detection.market_data import get_market_data
            from detection.smc_structure import detect_smc_structure
            
            md = get_market_data()
            tf = self.get_timeframe()
            klines = md.fetch_klines(symbol, limit=KLINES_LIMIT, interval=tf) \
                if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames \
                else md.fetch_klines(symbol, limit=KLINES_LIMIT)
            
            if not klines or len(klines) < 50:
                return {'symbol': symbol, 'error': 'Not enough klines', 'klines': [], 'analysis': {}}
            
            # On-demand chart fetch: use full klines (incl. live bar) so the
            # chart shows what TradingView would show in real time.
            analysis = detect_smc_structure(klines,
                                              internal_size=self.get_internal_size(),
                                              swing_size=int(self._settings.get('swing_size', 50)))
            
            with self._lock:
                self._cache[symbol] = {
                    'klines': klines,
                    'analysis': analysis,
                    'updated_at': time.time(),
                }
            
            return self._format_chart(symbol, klines, analysis)
        except Exception as e:
            return {'symbol': symbol, 'error': str(e), 'klines': [], 'analysis': {}}
    
    def _format_chart(self, symbol: str, klines: List[Dict], analysis: Dict) -> Dict:
        """Convert internal kline format to Lightweight Charts format."""
        ohlc = []
        for k in klines:
            t = k.get('t', 0)
            # Lightweight Charts expects timestamp in seconds (unix)
            t_sec = t // 1000 if t > 1e12 else t
            ohlc.append({
                'time': int(t_sec),
                'open': k.get('o', k.get('p', 0)),
                'high': k.get('h', k.get('p', 0)),
                'low': k.get('l', k.get('p', 0)),
                'close': k.get('p', 0),
                'volume': k.get('v', 0),
            })
        
        internal = analysis.get('internal', {})
        swing = analysis.get('swing', {})
        
        # Format pivots and events with timestamps in seconds
        def to_sec(t):
            if t is None:
                return 0
            return int(t // 1000) if t > 1e12 else int(t)
        
        def fmt_pivots(struct):
            return [{
                'time': to_sec(p.get('t')),
                'price': p.get('price'),
                'type': p.get('type'),
            } for p in struct.get('pivots', [])]
        
        def fmt_events(struct):
            return [{
                'from_time': to_sec(e.get('from_t')),
                'to_time': to_sec(e.get('to_t')),
                'level': e.get('level'),
                'tag': e.get('tag'),
                'dir': e.get('dir'),
            } for e in struct.get('events', [])]
        
        # === HTF Bias info (Tasks 1+2) ===
        # We always return ALL Internal events for the chart — user toggles
        # display of structure independently in the UI. The HTF bias is shown
        # in the trend badge and used as the watchlist dot color when the
        # filter is active. Signal markers (where alerts fired) are also
        # returned so the chart can plot LONG/SHORT dots at exact moments.
        htf_settings = self.get_htf_settings()
        with self._lock:
            htf_data = dict(self._htf_cache.get(symbol, {}))
            signals = list(self._signal_markers.get(symbol, []))
        htf_filter_active = htf_settings.get('enabled', False)
        htf_bias = htf_data.get('bias', 'neutral')
        
        all_events = fmt_events(internal)
        
        # Effective trend for the badge:
        #   When HTF filter is active and HTF has a clear direction, use it.
        #   Otherwise use Internal Structure's own trend.
        if htf_filter_active and htf_bias in ('bull', 'bear'):
            display_trend = 1 if htf_bias == 'bull' else -1
        else:
            display_trend = internal.get('trend', 0)
        
        # Strong High / Weak Low — the last swing pivot in the active trend dir,
        # marked as "strong" if the trend is going through it without breaking
        # opposite direction. For simplicity: just mark the most recent unbroken
        # swing high and low.
        strong_high = None
        weak_low = None
        if swing.get('pivots'):
            for p in reversed(swing['pivots']):
                if p['type'] in ('HH', 'LH') and strong_high is None:
                    strong_high = {
                        'time': to_sec(p.get('t')),
                        'price': p.get('price'),
                        'type': p.get('type'),
                    }
                if p['type'] in ('HL', 'LL') and weak_low is None:
                    weak_low = {
                        'time': to_sec(p.get('t')),
                        'price': p.get('price'),
                        'type': p.get('type'),
                    }
                if strong_high and weak_low:
                    break
        
        # === Forecast 1H + CTR (from forecast_engine cache) ===
        forecast_1h = None
        ctr = None
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe:
                cached_fc = fe.get(symbol)
                if cached_fc:
                    forecast_1h = cached_fc.get('forecast_1h')
                    ctr = cached_fc.get('ctr')
        except Exception:
            pass
        
        return {
            'symbol': symbol,
            'interval': self.get_display_label(),
            'ohlc': ohlc,
            # Internal Structure — ALL events, frontend toggles display
            'pivots': fmt_pivots(internal),
            'events': all_events,
            'trend': display_trend,
            # Signal markers — points where Telegram alerts actually fired
            'signals': signals,
            # HTF info for frontend display
            'htf_filter_active': htf_filter_active,
            'htf_bias': htf_bias,
            'htf_method': htf_data.get('method', ''),
            'htf_timeframe': htf_settings.get('timeframe', ''),
            # Forecast 1H + CTR (Pine PRO indicators)
            'forecast_1h': forecast_1h,
            'ctr': ctr,
            # Swing Structure (size=swing_size)
            'swing_pivots': fmt_pivots(swing),
            'swing_events': fmt_events(swing),
            'swing_trend': swing.get('trend', 0),
            'strong_high': strong_high,
            'weak_low': weak_low,
            'klines_count': len(ohlc),
            'updated_at': int(time.time()),
        }
    
    def get_state(self) -> Dict:
        with self._lock:
            htf_settings = self.get_htf_settings()
            htf_active = htf_settings.get('enabled', False)
            
            # Build per-symbol trend map
            # When HTF filter is active and that symbol has a clear HTF bias,
            # the watchlist dot reflects the HTF direction (matches what user
            # sees on the chart and what alerts will fire on).
            trends = {}
            for sym in self._watchlist:
                effective = 0
                if htf_active:
                    bias = self._htf_cache.get(sym, {}).get('bias', 'neutral')
                    if bias == 'bull':
                        effective = 1
                    elif bias == 'bear':
                        effective = -1
                    # neutral → fall through to internal trend
                if effective == 0:
                    c = self._cache.get(sym)
                    if c:
                        try:
                            effective = c.get('analysis', {}).get('internal', {}).get('trend', 0)
                        except:
                            effective = 0
                trends[sym] = effective
            
            # Per-symbol HTF biases
            htf_biases = {sym: dict(b) for sym, b in self._htf_cache.items()}
            
            return {
                'running': self._running,
                'enabled': self._settings.get('enabled', True),
                'watchlist': list(self._watchlist),
                'tradeable': list(self._tradeable),
                'settings': dict(self._settings),
                'scan_count': self._scan_count,
                'errors': self._errors,
                'cached_symbols': list(self._cache.keys()),
                'trends': trends,
                'htf_filter_active': htf_active,
                'htf_biases': htf_biases,
                'pending_choch': {k: {'dir': v['dir'], 'level': v['level']}
                                   for k, v in self._pending_choch.items()},
            }


# Singleton
_instance: Optional[SMCScanner] = None


def get_smc_scanner() -> Optional[SMCScanner]:
    return _instance


def init_smc_scanner(db=None, notifier=None) -> SMCScanner:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = SMCScanner(db=db, notifier=notifier)
    return _instance
