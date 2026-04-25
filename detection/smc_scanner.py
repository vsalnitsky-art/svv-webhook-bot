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
KLINES_LIMIT = 300            # bars to fetch per scan
TIMEFRAME = '15m'
MAX_WATCHLIST = 50

DB_KEY_WATCHLIST = 'smc_watchlist'
DB_KEY_SETTINGS = 'smc_settings'
DB_KEY_STATE = 'smc_last_events'   # tracks last seen event per symbol

DEFAULT_SETTINGS = {
    'enabled': True,
    'alert_mode': DEFAULT_ALERT_MODE,
    'interval_secs': DEFAULT_INTERVAL_SECS,
    'telegram_alerts': True,
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
        
        self._scan_count = 0
        self._errors = 0
        
        self._settings = self._load_settings()
        self._watchlist = self._load_watchlist()
    
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
    
    # ========================================
    # Public API — Watchlist
    # ========================================
    
    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self._watchlist)
    
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
                for k in list(self._last_alerted.keys()):
                    if k.startswith(f"{symbol}:"):
                        del self._last_alerted[k]
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
            allowed = ['enabled', 'alert_mode', 'interval_secs', 'telegram_alerts']
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
            
            self._persist_settings()
            return dict(self._settings)
    
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
        print(f"[SMC] ✅ Started: timeframe={TIMEFRAME}, "
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
                # Fetch 15m klines
                klines = md.fetch_klines(symbol, limit=KLINES_LIMIT, interval='15m') \
                    if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames \
                    else md.fetch_klines(symbol, limit=KLINES_LIMIT)
                
                if not klines or len(klines) < 50:
                    continue
                
                # Full result (includes live unclosed bar) — used for chart display
                result_full = detect_smc_structure(klines, internal_size=5)
                
                # Closed-bars-only result (drop last bar which is still forming)
                # — used for alert detection. This matches Pine's default
                # alertcondition() behavior which fires once per bar close.
                klines_closed = klines[:-1] if len(klines) > 1 else klines
                result_closed = detect_smc_structure(klines_closed, internal_size=5)
                
                # Cache for instant chart access (full data so user sees live bar)
                with self._lock:
                    self._cache[symbol] = {
                        'klines': klines,
                        'analysis': result_full,
                        'updated_at': time.time(),
                    }
                
                # Process events for alerts — use closed-bars-only result
                self._process_alerts(symbol, result_closed)
                
                # 200ms between symbols to spread load
                time.sleep(0.2)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[SMC] {symbol} scan error: {e}")
                self._errors += 1
        
        if self._scan_count <= 2 or self._scan_count % 30 == 0:
            print(f"[SMC] Scan #{self._scan_count}: {len(self._watchlist)} symbols, errors={self._errors}")
    
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
        # Only alert on events whose `to_t` (the bar where BOS/CHoCH was
        # confirmed by close cross) is within the LAST FEW closed bars.
        # This protects against alerts on events that "appeared" from history
        # due to algorithm re-evaluation.
        # 15m timeframe → allow events from last 3 closed bars (45 min).
        recent_threshold_secs = 60 * 60  # 1 hour cushion
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
            
            if mode == 'choch':
                if tag == 'CHoCH':
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
                            self._send_alert(symbol, ev, mode='choch_bos',
                                              choch_event=pending['choch_event'])
                            self._pending_choch.pop(symbol, None)
                    elif pending and pending['dir'] != ev['dir']:
                        # Opposite-direction BOS invalidates pending CHoCH
                        self._pending_choch.pop(symbol, None)
    
    def _send_alert(self, symbol: str, event: Dict, mode: str, choch_event: Dict = None):
        try:
            dir_icon = '🟢' if event['dir'] == 'bull' else '🔴'
            dir_label = 'BULLISH' if event['dir'] == 'bull' else 'BEARISH'
            
            level = event['level']
            level_str = f"${level:,.6g}" if level < 1 else f"${level:,.2f}"
            
            now_str = datetime.now(timezone.utc).strftime('%H:%M UTC')
            
            if mode == 'choch':
                msg = (
                    f"🔄 <b>CHoCH {dir_label}</b>: {symbol}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"{dir_icon} {dir_label} change of character\n"
                    f"📍 Level: {level_str}\n"
                    f"⏱ {TIMEFRAME} · {now_str}"
                )
            else:  # choch_bos
                choch_level = choch_event.get('level', 0) if choch_event else 0
                choch_str = f"${choch_level:,.6g}" if choch_level < 1 else f"${choch_level:,.2f}"
                msg = (
                    f"✅ <b>CHoCH+BOS confirmed</b>: {symbol}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"{dir_icon} {dir_label} structure confirmed\n"
                    f"🔄 CHoCH: {choch_str}\n"
                    f"💥 BOS: {level_str}\n"
                    f"⏱ {TIMEFRAME} · {now_str}"
                )
            
            self.notifier.send_message(msg)
            print(f"[SMC] 📨 Alert sent: {symbol} {mode} {dir_label}")
        except Exception as e:
            print(f"[SMC] Alert send error: {e}")
    
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
            klines = md.fetch_klines(symbol, limit=KLINES_LIMIT, interval='15m') \
                if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames \
                else md.fetch_klines(symbol, limit=KLINES_LIMIT)
            
            if not klines or len(klines) < 50:
                return {'symbol': symbol, 'error': 'Not enough klines', 'klines': [], 'analysis': {}}
            
            analysis = detect_smc_structure(klines, internal_size=5)
            
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
        
        # Format pivots and events with timestamps in seconds
        def to_sec(t):
            if t is None:
                return 0
            return int(t // 1000) if t > 1e12 else int(t)
        
        pivots = []
        for p in internal.get('pivots', []):
            pivots.append({
                'time': to_sec(p.get('t')),
                'price': p.get('price'),
                'type': p.get('type'),
            })
        
        events = []
        for e in internal.get('events', []):
            events.append({
                'from_time': to_sec(e.get('from_t')),
                'to_time': to_sec(e.get('to_t')),
                'level': e.get('level'),
                'tag': e.get('tag'),
                'dir': e.get('dir'),
            })
        
        return {
            'symbol': symbol,
            'interval': TIMEFRAME,
            'ohlc': ohlc,
            'pivots': pivots,
            'events': events,
            'trend': internal.get('trend', 0),
            'klines_count': len(ohlc),
            'updated_at': int(time.time()),
        }
    
    def get_state(self) -> Dict:
        with self._lock:
            # Build per-symbol trend map from cache
            trends = {}
            for sym, c in self._cache.items():
                try:
                    trends[sym] = c.get('analysis', {}).get('internal', {}).get('trend', 0)
                except:
                    trends[sym] = 0
            
            return {
                'running': self._running,
                'enabled': self._settings.get('enabled', True),
                'watchlist': list(self._watchlist),
                'settings': dict(self._settings),
                'scan_count': self._scan_count,
                'errors': self._errors,
                'cached_symbols': list(self._cache.keys()),
                'trends': trends,
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
