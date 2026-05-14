"""Volumized OB Radar — TOP-N scanner for fresh Volumized OBs in P/D zones.

Mirrors the architecture of `top100_ob_scanner.py`, but uses the Pine
Volumized OB algorithm (from `volumized_ob.py`) instead of the simpler
OB detector, and applies a Premium/Discount zone filter (Pine notion):

  • BULL OB qualifies only when its mid-point sits below 38.2% of the
    current swing range (Discount zone — favorable for LONG entry).
  • BEAR OB qualifies only when its mid-point sits above 61.8% of the
    current swing range (Premium zone — favorable for SHORT entry).
  • Equilibrium zone (38.2-61.8%) → skipped, not added.

Lifecycle of a radar-added symbol:

  Scan tick → qualifying OB found
              ↓
  Symbol already in watchlist?  yes → skip silently (action='skipped_already_in_watchlist')
              ↓ no
  Symbol on cooldown after recent removal?  yes → skip (action='skipped_cooldown')
              ↓ no
  Radar at concurrent-items capacity?  yes → skip (action='skipped_capacity_full')
              ↓ no
  Add to SMC watchlist via smc_scanner.add_symbol() + insert metadata row
  with 24h TTL + bump stats.times_added + telegram alert.
              ↓
  (Daemon background loop, every scan tick:)
              ↓
  ─── Path A: SMC signal fires within 24h ───
      smc_scanner._send_alert → db.volradar_mark_signal_fired() →
      metadata row deleted, stats.times_signal_fired++,
      symbol stays in watchlist as a normal item.
              
  ─── Path B: 24h elapses, no signal ───
      Cleanup loop calls volradar_find_expired() → for each expired
      symbol: remove from watchlist via smc_scanner.remove_symbol() +
      delete metadata row + stats.times_auto_removed++ + cooldown set.
              
  ─── Path C: user manually removes from watchlist ───
      Existing /api/smc/remove already calls smc_scanner.remove_symbol().
      We hook into that flow elsewhere (NOT in this file) to ensure
      stats.times_manual_removed++ + cooldown set on manual removes.

NOT a signal generator. Doesn't fire trades, doesn't drive Trade Manager
directly. Mode A only: add to watchlist, let SMC scanner handle entries
through its normal signal path. The radar's value is *pre-filtering*
attention — only watch coins with fresh, well-positioned OBs.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from detection.market_data import get_market_data
from detection.volumized_ob import detect_volumized_obs
from storage.db_operations import get_db


# === Tuneable defaults — overridable via settings UI ===
DEFAULT_TOP_N = 100
DEFAULT_MIN_QUOTE_VOLUME_USD = 100_000_000   # $100M (matches TOP-100 default)
DEFAULT_THROTTLE_MS = 600
KLINES_LIMIT = 2000          # Volumized needs deep history (matches main scanner)
DEFAULT_SCAN_TIMEFRAME = '1h'
ALLOWED_SCAN_TIMEFRAMES = ('15m', '30m', '1h', '2h', '4h')
DEFAULT_SCAN_INTERVAL_MIN = 10

# Volumized algorithm params — independent from main SMC settings by design.
# User can hit "Copy from main settings" button in the UI to mirror.
DEFAULT_SWING_LENGTH = 10               # Pine default (stricter than main's 5)
DEFAULT_OB_END_METHOD = 'Wick'
DEFAULT_MAX_ATR_MULT = 3.5
DEFAULT_ZONE_COUNT = 'Low'
DEFAULT_COMBINE_OBS = True

# Premium/Discount thresholds — Pine golden-ratio Fibonacci (38.2 / 61.8).
# OB qualifies if its mid sits OUTSIDE the equilibrium band (38.2-61.8).
DEFAULT_DISCOUNT_THRESHOLD = 38.2
DEFAULT_PREMIUM_THRESHOLD = 61.8

# Lifecycle params
DEFAULT_TTL_HOURS = 24                  # User-locked: 24h then auto-remove
DEFAULT_COOLDOWN_HOURS = 6              # Block re-adds for this long after removal
DEFAULT_MAX_CONCURRENT_ITEMS = 20       # FIFO cap on radar-added items in flight
DEFAULT_FRESH_WINDOW_HOURS = 6          # OB must be ≤ this old to qualify

# Settings DB key (one JSON blob — same convention as TOP-100)
SETTINGS_KEY = 'volumized_radar_settings'


class VolumizedOBRadar:
    """Background daemon — scans top-N USDT-perpetuals every N minutes,
    looks for fresh Volumized OBs in P/D zones, auto-adds them to the
    SMC watchlist with 24h TTL. Singleton via get_volumized_ob_radar()."""
    
    def __init__(self, telegram_notifier=None):
        self.md = get_market_data()
        self.db = get_db()
        self.tg = telegram_notifier
        
        # Defaults — get overlaid by _load_settings() below
        self._enabled = False
        self._top_n = DEFAULT_TOP_N
        self._min_quote_volume_usd = DEFAULT_MIN_QUOTE_VOLUME_USD
        self._throttle_ms = DEFAULT_THROTTLE_MS
        self._timeframe = DEFAULT_SCAN_TIMEFRAME
        self._scan_interval_min = DEFAULT_SCAN_INTERVAL_MIN
        self._fresh_window_hours = DEFAULT_FRESH_WINDOW_HOURS
        self._ttl_hours = DEFAULT_TTL_HOURS
        self._cooldown_hours = DEFAULT_COOLDOWN_HOURS
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_ITEMS
        self._discount_threshold = DEFAULT_DISCOUNT_THRESHOLD
        self._premium_threshold = DEFAULT_PREMIUM_THRESHOLD
        
        # Volumized algorithm params (independent from main settings)
        self._swing_length = DEFAULT_SWING_LENGTH
        self._ob_end_method = DEFAULT_OB_END_METHOD
        self._max_atr_mult = DEFAULT_MAX_ATR_MULT
        self._zone_count = DEFAULT_ZONE_COUNT
        self._combine_obs = DEFAULT_COMBINE_OBS
        
        # Daemon state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_scan_at: Optional[datetime] = None
        self._last_scan_summary: Dict = {}
        self._scan_lock = threading.Lock()
        
        self._load_settings()
    
    # ============================================================
    # Settings persistence
    # ============================================================
    
    def _load_settings(self):
        """Load settings from DB. Falls back to defaults if missing/corrupt.
        Called at startup AND can be re-called after update_settings to
        re-read (though update_settings updates in-memory directly)."""
        raw = self.db.get_setting(SETTINGS_KEY)
        if not raw:
            return
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(data, dict):
                return
            self._enabled = bool(data.get('enabled', self._enabled))
            self._top_n = int(data.get('top_n', self._top_n))
            self._min_quote_volume_usd = float(data.get('min_quote_volume_usd', self._min_quote_volume_usd))
            self._throttle_ms = int(data.get('throttle_ms', self._throttle_ms))
            tf = data.get('timeframe', self._timeframe)
            if tf in ALLOWED_SCAN_TIMEFRAMES:
                self._timeframe = tf
            self._scan_interval_min = int(data.get('scan_interval_min', self._scan_interval_min))
            self._fresh_window_hours = int(data.get('fresh_window_hours', self._fresh_window_hours))
            self._ttl_hours = int(data.get('ttl_hours', self._ttl_hours))
            self._cooldown_hours = int(data.get('cooldown_hours', self._cooldown_hours))
            self._max_concurrent = int(data.get('max_concurrent_items', self._max_concurrent))
            self._discount_threshold = float(data.get('discount_threshold', self._discount_threshold))
            self._premium_threshold = float(data.get('premium_threshold', self._premium_threshold))
            self._swing_length = int(data.get('swing_length', self._swing_length))
            self._ob_end_method = str(data.get('ob_end_method', self._ob_end_method))
            self._max_atr_mult = float(data.get('max_atr_mult', self._max_atr_mult))
            self._zone_count = str(data.get('zone_count', self._zone_count))
            self._combine_obs = bool(data.get('combine_obs', self._combine_obs))
        except Exception as e:
            print(f'[VOLRADAR] settings load error: {e}')
    
    def _save_settings(self):
        """Persist current in-memory state to DB."""
        try:
            self.db.set_setting(SETTINGS_KEY, json.dumps(self.get_settings()))
        except Exception as e:
            print(f'[VOLRADAR] settings save error: {e}')
    
    def get_settings(self) -> Dict:
        """Return all settings as a plain dict (used by /api/vol-radar/settings
        GET endpoint and the UI form)."""
        return {
            'enabled': self._enabled,
            'top_n': self._top_n,
            'min_quote_volume_usd': self._min_quote_volume_usd,
            'throttle_ms': self._throttle_ms,
            'timeframe': self._timeframe,
            'scan_interval_min': self._scan_interval_min,
            'fresh_window_hours': self._fresh_window_hours,
            'ttl_hours': self._ttl_hours,
            'cooldown_hours': self._cooldown_hours,
            'max_concurrent_items': self._max_concurrent,
            'discount_threshold': self._discount_threshold,
            'premium_threshold': self._premium_threshold,
            'swing_length': self._swing_length,
            'ob_end_method': self._ob_end_method,
            'max_atr_mult': self._max_atr_mult,
            'zone_count': self._zone_count,
            'combine_obs': self._combine_obs,
        }
    
    def update_settings(self, **kwargs) -> Dict:
        """Merge in new settings, validate, persist, and return current state.
        Unknown keys are silently ignored; the API endpoint should pre-filter
        to keep the contract clean."""
        if 'enabled' in kwargs:
            self._enabled = bool(kwargs['enabled'])
        if 'top_n' in kwargs:
            v = int(kwargs['top_n'])
            if 10 <= v <= 500:
                self._top_n = v
        if 'min_quote_volume_usd' in kwargs:
            self._min_quote_volume_usd = float(kwargs['min_quote_volume_usd'])
        if 'throttle_ms' in kwargs:
            v = int(kwargs['throttle_ms'])
            if 100 <= v <= 5000:
                self._throttle_ms = v
        if 'timeframe' in kwargs:
            tf = kwargs['timeframe']
            if tf in ALLOWED_SCAN_TIMEFRAMES:
                self._timeframe = tf
        if 'scan_interval_min' in kwargs:
            v = int(kwargs['scan_interval_min'])
            if 1 <= v <= 240:
                self._scan_interval_min = v
        if 'fresh_window_hours' in kwargs:
            v = int(kwargs['fresh_window_hours'])
            if 1 <= v <= 168:
                self._fresh_window_hours = v
        if 'ttl_hours' in kwargs:
            v = int(kwargs['ttl_hours'])
            if 1 <= v <= 168:
                self._ttl_hours = v
        if 'cooldown_hours' in kwargs:
            v = int(kwargs['cooldown_hours'])
            if 0 <= v <= 168:
                self._cooldown_hours = v
        if 'max_concurrent_items' in kwargs:
            v = int(kwargs['max_concurrent_items'])
            if 1 <= v <= 100:
                self._max_concurrent = v
        if 'discount_threshold' in kwargs:
            v = float(kwargs['discount_threshold'])
            if 0 < v < 50:
                self._discount_threshold = v
        if 'premium_threshold' in kwargs:
            v = float(kwargs['premium_threshold'])
            if 50 < v < 100:
                self._premium_threshold = v
        # Volumized params (mirror keys from main settings for "copy" support)
        if 'swing_length' in kwargs:
            v = int(kwargs['swing_length'])
            if 3 <= v <= 50:
                self._swing_length = v
        if 'ob_end_method' in kwargs and kwargs['ob_end_method'] in ('Wick', 'Close'):
            self._ob_end_method = kwargs['ob_end_method']
        if 'max_atr_mult' in kwargs:
            v = float(kwargs['max_atr_mult'])
            if 0.5 <= v <= 20.0:
                self._max_atr_mult = v
        if 'zone_count' in kwargs and kwargs['zone_count'] in ('One', 'Low', 'Medium', 'High'):
            self._zone_count = kwargs['zone_count']
        if 'combine_obs' in kwargs:
            self._combine_obs = bool(kwargs['combine_obs'])
        
        self._save_settings()
        return self.get_settings()
    
    def copy_from_main_volumized(self) -> Dict:
        """Pull Volumized algorithm params from the SMC scanner's main
        settings — convenience for "make radar use same OB detection
        flavor as main scanner". UI exposes this as a single button."""
        try:
            from detection.smc_scanner import get_smc_scanner
            sc = get_smc_scanner()
            main = sc._settings
            self.update_settings(
                swing_length=main.get('volumized_swing_length', self._swing_length),
                ob_end_method=main.get('volumized_ob_end_method', self._ob_end_method),
                max_atr_mult=main.get('volumized_max_atr_mult', self._max_atr_mult),
                zone_count=main.get('volumized_zone_count', self._zone_count),
                combine_obs=main.get('volumized_combine_obs', self._combine_obs),
                timeframe=main.get('volumized_timeframe', self._timeframe),
            )
            return self.get_settings()
        except Exception as e:
            print(f'[VOLRADAR] copy_from_main error: {e}')
            return self.get_settings()
    
    # ============================================================
    # Daemon control
    # ============================================================
    
    def start(self):
        """Spawn daemon thread. Idempotent — already-running call is a no-op."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop,
                                         name='VolumizedOBRadar', daemon=True)
        self._thread.start()
        print('[VOLRADAR] Daemon started')
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print('[VOLRADAR] Daemon stopped')
    
    def _scheduler_loop(self):
        """Main daemon loop. Runs scan + cleanup on schedule. Uses a short
        sleep granularity (5s) so we can respond to stop() quickly without
        waiting out a 10-minute interval."""
        next_scan_at = datetime.utcnow()   # immediate first scan
        next_prune_at = datetime.utcnow() + timedelta(hours=24)
        
        while not self._stop_event.is_set():
            now = datetime.utcnow()
            
            # Scheduled scan tick
            if self._enabled and now >= next_scan_at:
                try:
                    self.scan(triggered_by='schedule')
                except Exception as e:
                    print(f'[VOLRADAR] scheduled scan error: {e}')
                next_scan_at = now + timedelta(minutes=self._scan_interval_min)
            
            # Daily audit-log prune (cheap, runs even when disabled)
            if now >= next_prune_at:
                try:
                    deleted = self.db.volradar_prune_snapshots(retention_days=7)
                    if deleted > 0:
                        print(f'[VOLRADAR] pruned {deleted} old snapshot rows')
                except Exception as e:
                    print(f'[VOLRADAR] prune error: {e}')
                next_prune_at = now + timedelta(hours=24)
            
            # Short sleep — granularity for stop() responsiveness
            self._stop_event.wait(timeout=5)
    
    # ============================================================
    # Scanning
    # ============================================================
    
    def scan(self, triggered_by: str = 'manual') -> Dict:
        """Run one full scan tick. Returns a summary dict. Thread-safe via
        scan_lock — concurrent manual+scheduled calls serialize."""
        with self._scan_lock:
            return self._do_scan(triggered_by)
    
    def _do_scan(self, triggered_by: str) -> Dict:
        started_at = datetime.utcnow()
        
        # === 1. Cleanup expired items FIRST (before adding new ones) ===
        # This frees up capacity if we're near the cap, and ensures the
        # snapshot we take below reflects post-cleanup state.
        expired_removed = self._cleanup_expired_items()
        
        # === 2. Fetch universe ===
        try:
            universe = self.md.fetch_top_perp_symbols(
                n=self._top_n,
                min_quote_volume_usd=self._min_quote_volume_usd)
        except Exception as e:
            print(f'[VOLRADAR] fetch_top_perp_symbols failed: {e}')
            universe = None
        
        if not universe:
            summary = {
                'status': 'error',
                'triggered_by': triggered_by,
                'started_at': started_at.isoformat(),
                'finished_at': datetime.utcnow().isoformat(),
                'message': 'Failed to fetch universe',
            }
            self._last_scan_summary = summary
            self._last_scan_at = started_at
            return summary
        
        print(f'[VOLRADAR] Universe: {len(universe)} symbols, scan_tf={self._timeframe}')
        
        # === 3. Snapshot current watchlist for dedup checks ===
        try:
            from detection.smc_scanner import get_smc_scanner
            scanner = get_smc_scanner()
            current_watchlist = set(scanner.get_watchlist())
        except Exception as e:
            print(f'[VOLRADAR] watchlist snapshot error: {e}')
            current_watchlist = set()
        
        # Current radar item count (for capacity check)
        current_radar_items = self.db.volradar_list_metadata()
        radar_count = len(current_radar_items)
        
        # === 4. Per-symbol scan ===
        symbols_scanned = 0
        ob_qualified = 0
        added_count = 0
        skipped_existing = 0
        skipped_cooldown = 0
        skipped_capacity = 0
        skipped_not_in_zone = 0
        skipped_too_old = 0
        skipped_no_swings = 0
        added_symbols: List[Dict] = []
        errors = 0
        
        for entry in universe:
            if self._stop_event.is_set():
                break
            symbol = entry['symbol']
            try:
                scan_result = self._scan_symbol(symbol)
                symbols_scanned += 1
                
                if scan_result is None:
                    continue   # already logged inside _scan_symbol
                
                ob_info = scan_result['ob_info']
                qualifies = scan_result['qualifies']
                
                if not qualifies:
                    # Refine in-memory counter (was always bumping
                    # skipped_not_in_zone even on too-old / no-swings).
                    skip_reason = scan_result.get('skip_reason', 'skipped_not_in_zone')
                    if skip_reason == 'skipped_ob_too_old':
                        skipped_too_old += 1
                    elif skip_reason == 'skipped_no_swings':
                        skipped_no_swings += 1
                    else:
                        skipped_not_in_zone += 1
                    
                    # Build a diagnostic line that lands in the Snapshots
                    # panel's error_msg column. Surfaces WHY we rejected:
                    #   - skipped_ob_too_old      → "age=4.2h (limit 1h)"
                    #   - skipped_not_in_zone     → "BULL pd=52% (need ≤45)"
                    #     or                      → "BEAR pd=48% (need ≥55)"
                    #   - skipped_no_swings       → "no swings formed yet"
                    diag_msg = None
                    if ob_info and skip_reason == 'skipped_ob_too_old':
                        diag_msg = (f"age={ob_info.get('age_hours','?')}h "
                                    f"(limit {self._fresh_window_hours}h)")
                    elif ob_info and skip_reason == 'skipped_not_in_zone':
                        d = ob_info.get('direction', '?')
                        p = ob_info.get('pd_zone_pct')
                        if d == 'BULL':
                            diag_msg = (f"BULL pd={p}% "
                                        f"(need ≤{self._discount_threshold})")
                        elif d == 'BEAR':
                            diag_msg = (f"BEAR pd={p}% "
                                        f"(need ≥{self._premium_threshold})")
                    elif skip_reason == 'skipped_no_swings':
                        diag_msg = 'no swings formed yet'
                    
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=False,
                        action=skip_reason,
                        ob_direction=ob_info.get('direction') if ob_info else None,
                        ob_top=ob_info.get('top') if ob_info else None,
                        ob_bottom=ob_info.get('bottom') if ob_info else None,
                        pd_zone_pct=ob_info.get('pd_zone_pct') if ob_info else None,
                        error_msg=diag_msg,
                    )
                    continue
                
                ob_qualified += 1
                
                # === Qualifying logic — duplicate + cooldown + capacity ===
                if symbol in current_watchlist:
                    skipped_existing += 1
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=True,
                        action='skipped_already_in_watchlist',
                        ob_direction=ob_info['direction'],
                        ob_top=ob_info['top'], ob_bottom=ob_info['bottom'],
                        pd_zone_pct=ob_info['pd_zone_pct'],
                    )
                    continue
                
                if self.db.volradar_is_on_cooldown(symbol):
                    skipped_cooldown += 1
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=True,
                        action='skipped_cooldown',
                        ob_direction=ob_info['direction'],
                        ob_top=ob_info['top'], ob_bottom=ob_info['bottom'],
                        pd_zone_pct=ob_info['pd_zone_pct'],
                    )
                    continue
                
                if radar_count >= self._max_concurrent:
                    skipped_capacity += 1
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=True,
                        action='skipped_capacity_full',
                        ob_direction=ob_info['direction'],
                        ob_top=ob_info['top'], ob_bottom=ob_info['bottom'],
                        pd_zone_pct=ob_info['pd_zone_pct'],
                    )
                    continue
                
                # === All checks passed — add to watchlist + radar metadata ===
                add_ok = self._add_to_watchlist_with_metadata(symbol, ob_info)
                if add_ok:
                    added_count += 1
                    radar_count += 1
                    current_watchlist.add(symbol)
                    added_symbols.append({'symbol': symbol, **ob_info})
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=True, action='added',
                        ob_direction=ob_info['direction'],
                        ob_top=ob_info['top'], ob_bottom=ob_info['bottom'],
                        pd_zone_pct=ob_info['pd_zone_pct'],
                    )
                else:
                    self.db.volradar_log_snapshot(
                        symbol=symbol, qualified=True, action='error',
                        ob_direction=ob_info['direction'],
                        ob_top=ob_info['top'], ob_bottom=ob_info['bottom'],
                        pd_zone_pct=ob_info['pd_zone_pct'],
                        error_msg='watchlist add failed',
                    )
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f'[VOLRADAR] Error scanning {symbol}: {e}')
                self.db.volradar_log_snapshot(
                    symbol=symbol, qualified=False, action='error',
                    error_msg=str(e)[:200],
                )
            
            # Throttle between API calls (kind to Binance)
            time.sleep(self._throttle_ms / 1000.0)
        
        finished_at = datetime.utcnow()
        duration_sec = (finished_at - started_at).total_seconds()
        
        summary = {
            'status': 'ok',
            'triggered_by': triggered_by,
            'started_at': started_at.isoformat(),
            'finished_at': finished_at.isoformat(),
            'duration_sec': round(duration_sec, 1),
            'symbols_scanned': symbols_scanned,
            'ob_qualified': ob_qualified,
            'added': added_count,
            'skipped_already_in_watchlist': skipped_existing,
            'skipped_cooldown': skipped_cooldown,
            'skipped_capacity_full': skipped_capacity,
            'skipped_not_in_zone': skipped_not_in_zone,
            'skipped_too_old': skipped_too_old,
            'skipped_no_swings': skipped_no_swings,
            'expired_removed': expired_removed,
            'errors': errors,
            'added_symbols': added_symbols,
        }
        self._last_scan_summary = summary
        self._last_scan_at = started_at
        
        print(f'[VOLRADAR] Scan done in {duration_sec:.1f}s: '
              f'{added_count} added, {expired_removed} expired, '
              f'{ob_qualified} qualified, {symbols_scanned} scanned '
              f'| skip: zone={skipped_not_in_zone} old={skipped_too_old} '
              f'noswing={skipped_no_swings} cap={skipped_capacity} '
              f'cooldown={skipped_cooldown} existing={skipped_existing}')
        
        # Telegram alert for adds
        if added_symbols:
            try:
                self._notify_telegram_adds(added_symbols)
            except Exception as e:
                print(f'[VOLRADAR] telegram error: {e}')
        
        return summary
    
    def _scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Run Volumized OB detection on one symbol. Returns:
            {'ob_info': {...} | None,
             'qualifies': bool,
             'skip_reason': str}  # only when qualifies=False
        OR None when scan was unable to run (and logged a snapshot row).
        """
        klines = self.md.fetch_klines(symbol, limit=KLINES_LIMIT,
                                       interval=self._timeframe)
        if not klines or len(klines) < self._swing_length + 50:
            self.db.volradar_log_snapshot(
                symbol=symbol, qualified=False, action='skipped_no_ob',
                error_msg=f'insufficient klines ({len(klines) if klines else 0})',
            )
            return None
        
        result = detect_volumized_obs(
            klines,
            swing_length=self._swing_length,
            ob_end_method=self._ob_end_method,
            max_atr_mult=self._max_atr_mult,
            zone_count=self._zone_count,
            combine_obs=self._combine_obs,
        )
        
        latest = result.get('latest_ob')
        if latest is None:
            self.db.volradar_log_snapshot(
                symbol=symbol, qualified=False, action='skipped_no_ob',
            )
            return None
        
        swing_high = result.get('swing_high')
        swing_low = result.get('swing_low')
        if swing_high is None or swing_low is None or swing_high <= swing_low:
            return {'ob_info': None, 'qualifies': False, 'skip_reason': 'skipped_no_swings'}
        
        # === Freshness check ===
        # IMPORTANT: use formation_time (the bar where the OB was TRIGGERED
        # by close crossing the swing), NOT start_time (the wick-anchor bar
        # we walked back to). The latter can be many bars before formation
        # — on 1H with swing_length=8, anchors typically sit 5-30 hours back,
        # so checking anchor age against a 1-6h fresh window rejects newly
        # formed OBs that are still actionable. Backward-compat fallback to
        # start_time when formation_time is absent (older OB dicts in tests).
        ob_start_ms = latest.get('start_time', 0)          # anchor bar (for display)
        ob_formation_ms = latest.get('formation_time') or ob_start_ms  # trigger bar (for freshness)
        ob_age_hours = (time.time() * 1000 - ob_formation_ms) / 3_600_000.0
        if ob_age_hours > self._fresh_window_hours:
            return {
                'ob_info': {
                    'direction': 'BULL' if latest['type'] == 'Bull' else 'BEAR',
                    'top': latest['top'], 'bottom': latest['bottom'],
                    'volume': latest.get('ob_volume', 0.0),
                    'pd_zone_pct': None,
                    'age_hours': round(ob_age_hours, 2),
                },
                'qualifies': False,
                'skip_reason': 'skipped_ob_too_old',
            }
        
        # === P/D zone calculation ===
        # OB mid-point as % of swing range. 0% = swing_low, 100% = swing_high.
        ob_mid = (latest['top'] + latest['bottom']) / 2.0
        swing_range = swing_high - swing_low
        pd_pct = ((ob_mid - swing_low) / swing_range) * 100.0
        
        is_bull = (latest['type'] == 'Bull')
        ob_info = {
            'direction': 'BULL' if is_bull else 'BEAR',
            'top': latest['top'],
            'bottom': latest['bottom'],
            'volume': latest.get('ob_volume', 0.0),
            'pd_zone_pct': round(pd_pct, 2),
            'age_hours': round(ob_age_hours, 2),
            'swing_high': swing_high,
            'swing_low': swing_low,
            'start_time_ms': ob_start_ms,
        }
        
        # === Qualifying decision ===
        # BULL only in Discount (≤ discount_threshold)
        # BEAR only in Premium (≥ premium_threshold)
        qualifies = False
        if is_bull and pd_pct <= self._discount_threshold:
            qualifies = True
        elif (not is_bull) and pd_pct >= self._premium_threshold:
            qualifies = True
        
        return {
            'ob_info': ob_info,
            'qualifies': qualifies,
            'skip_reason': 'skipped_not_in_zone' if not qualifies else '',
        }
    
    # ============================================================
    # Watchlist add (Mode A only — no direct Trade Manager call)
    # ============================================================
    
    def _add_to_watchlist_with_metadata(self, symbol: str, ob_info: Dict) -> bool:
        """Add symbol to SMC watchlist via scanner's API + insert radar
        metadata row + stats bump. Returns True on full success."""
        try:
            from detection.smc_scanner import get_smc_scanner
            scanner = get_smc_scanner()
            add_result = scanner.add_symbol(symbol)
            if not add_result.get('ok'):
                # Likely "Already in watchlist" or "Max symbols" — log and skip
                print(f'[VOLRADAR] add_symbol({symbol}) failed: {add_result.get("reason")}')
                return False
        except Exception as e:
            print(f'[VOLRADAR] scanner.add_symbol error for {symbol}: {e}')
            return False
        
        # Insert metadata row (idempotent — if exists, returns False but
        # symbol is already in watchlist which is fine).
        meta_ok = self.db.volradar_add(
            symbol=symbol,
            ob_direction=ob_info['direction'],
            ob_top=ob_info['top'],
            ob_bottom=ob_info['bottom'],
            ob_volume=ob_info['volume'],
            pd_zone_pct=ob_info['pd_zone_pct'],
            scan_tf=self._timeframe,
            ttl_hours=self._ttl_hours,
        )
        if not meta_ok:
            print(f'[VOLRADAR] metadata insert returned False for {symbol} (may already exist)')
        
        return True
    
    # ============================================================
    # Cleanup expired items
    # ============================================================
    
    def _cleanup_expired_items(self) -> int:
        """Remove watchlist symbols whose 24h TTL elapsed. Each removal:
        1. Calls smc_scanner.remove_symbol() — fully detaches from scanner state
        2. db.volradar_remove(reason='auto_ttl') — bumps counter + sets cooldown
        Returns count of removals."""
        expired = self.db.volradar_find_expired()
        if not expired:
            return 0
        
        removed = 0
        try:
            from detection.smc_scanner import get_smc_scanner
            scanner = get_smc_scanner()
        except Exception as e:
            print(f'[VOLRADAR] cleanup scanner-import error: {e}')
            return 0
        
        for symbol in expired:
            try:
                scanner.remove_symbol(symbol)
                if self.db.volradar_remove(symbol, reason='auto_ttl',
                                            cooldown_hours=self._cooldown_hours):
                    removed += 1
                    print(f'[VOLRADAR] {symbol} auto-removed (24h TTL expired, '
                          f'cooldown {self._cooldown_hours}h)')
            except Exception as e:
                print(f'[VOLRADAR] cleanup error for {symbol}: {e}')
        return removed
    
    # ============================================================
    # External hooks (called from API endpoints / other modules)
    # ============================================================
    
    def manual_remove(self, symbol: str) -> Dict:
        """User clicked 'Remove' on a radar-tracked symbol. Wraps scanner
        removal + bumps manual counter + sets cooldown. Idempotent."""
        try:
            from detection.smc_scanner import get_smc_scanner
            scanner = get_smc_scanner()
            scanner.remove_symbol(symbol)
            removed = self.db.volradar_remove(symbol, reason='manual',
                                               cooldown_hours=self._cooldown_hours)
            return {'ok': True, 'symbol': symbol, 'metadata_removed': removed}
        except Exception as e:
            return {'ok': False, 'reason': str(e)}
    
    def get_state(self) -> Dict:
        """Dashboard state — settings + active items + recent snapshots + stats.
        Single-call shape used by the radar UI panel."""
        return {
            'settings': self.get_settings(),
            'last_scan_at': self._last_scan_at.isoformat() if self._last_scan_at else None,
            'last_scan_summary': self._last_scan_summary,
            'active_items': self.db.volradar_list_metadata(),
            'recent_snapshots': self.db.volradar_list_snapshots(hours=24, limit=200),
            'top_stats': self.db.volradar_list_stats(limit=50),
            'daemon_running': bool(self._thread and self._thread.is_alive()),
        }
    
    # ============================================================
    # Telegram alerts
    # ============================================================
    
    def _notify_telegram_adds(self, added: List[Dict]) -> None:
        """Send one consolidated message per scan (avoids spam if many adds)."""
        if not self.tg or not added:
            return
        
        lines = ['📦 *Volumized OB Radar — auto-add*\n']
        for item in added:
            sym = item['symbol']
            d = item['direction']
            top = item['top']
            bot = item['bottom']
            pd = item['pd_zone_pct']
            arrow = '🟢 Bull' if d == 'BULL' else '🔴 Bear'
            zone = 'Discount' if d == 'BULL' else 'Premium'
            lines.append(
                f"• `{sym}` — Fresh {arrow} OB at "
                f"`{self._fmt_price(bot)} – {self._fmt_price(top)}` "
                f"({zone} {pd:.1f}%, Vol {self._timeframe.upper()})"
            )
        lines.append(f"\nAdded to watchlist (TTL {self._ttl_hours}h).")
        
        msg = '\n'.join(lines)
        try:
            self.tg.send_message(msg, parse_mode='Markdown')
        except Exception as e:
            # Markdown might fail on special characters in symbols — try plain
            try:
                self.tg.send_message(msg)
            except Exception as e2:
                print(f'[VOLRADAR] telegram send failed: {e} / {e2}')
    
    @staticmethod
    def _fmt_price(p: float) -> str:
        """Compact price formatter — different precision per magnitude."""
        if p is None:
            return '?'
        if p >= 1000:
            return f'{p:,.2f}'
        elif p >= 1:
            return f'{p:.4f}'
        elif p >= 0.01:
            return f'{p:.6f}'
        else:
            return f'{p:.8f}'


# ============================================================
# Singleton
# ============================================================
_instance: Optional[VolumizedOBRadar] = None


def get_volumized_ob_radar(telegram_notifier=None) -> VolumizedOBRadar:
    """Singleton accessor. First call may set telegram_notifier; subsequent
    calls return the same instance (telegram_notifier is ignored on those)."""
    global _instance
    if _instance is None:
        _instance = VolumizedOBRadar(telegram_notifier=telegram_notifier)
    return _instance
