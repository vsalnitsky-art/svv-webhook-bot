"""TOP-100 4H OB Radar — Variant B (scheduled scan).

Periodically scans the top-N USDT-perpetual symbols on Binance Futures
for newly-formed 4H Order Blocks using the Pine SMC_PRO_BOT__47_
algorithm (same detector as the main SMC scanner — `ob_detector.py`).

Architecture:
- Daemon thread, runs schedule + on-demand triggers.
- Schedule: every 4 hours at :01 (gives 4H bar a minute to close on Binance).
- On-demand: API endpoint triggers a manual scan.
- Throttled: 600ms between API calls — full scan ~60s for 100 symbols.
- Two outputs: DB snapshot (current state) + history audit log.
- Telegram alerts: only on FRESH CHoCH-created OBs (the strict-mode
  setup — same gate logic as the main SMC scanner's OB Filter).

NOT a signal generator. Purely informational. Doesn't affect SMC scanner
or trade manager. User can opt-in to "Add to SMC Watchlist" per symbol
via the UI, which is what bridges this scanner into the trading path.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

from detection.market_data import get_market_data
from detection.smc_structure import detect_smc_structure
from detection.ob_detector import detect_last_order_block
from storage.db_operations import get_db


# Tuneable defaults — overridable via settings
DEFAULT_TOP_N = 100
DEFAULT_MIN_QUOTE_VOLUME_USD = 100_000_000  # $100M default per spec
DEFAULT_THROTTLE_MS = 600
KLINES_LIMIT = 400          # ATR-200 needs at least 200 + buffer
SCAN_TIMEFRAME = '4h'
SCAN_INTERVAL_HOURS = 4     # Schedule cadence
HISTORY_RETENTION_DAYS = 30


class Top100OBScanner:
    """Daemon-backed scanner for TOP-N 4H OBs."""
    
    def __init__(self, telegram_notifier=None):
        self.md = get_market_data()
        self.db = get_db()
        self._notifier = telegram_notifier
        
        # Settings (loadable via setter)
        self._enabled = False
        self._telegram_alerts_enabled = True
        self._include_bos_alerts = False  # Default: only CHoCH alerts
        self._min_quote_volume_usd = DEFAULT_MIN_QUOTE_VOLUME_USD
        self._top_n = DEFAULT_TOP_N
        self._throttle_ms = DEFAULT_THROTTLE_MS
        
        # Runtime state
        self._scan_lock = threading.Lock()  # Prevents concurrent scans
        self._is_scanning = False
        self._last_scan_at: Optional[datetime] = None
        self._last_scan_summary: Dict = {}
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
    
    # ============================================================
    # Settings interface
    # ============================================================
    
    def update_settings(self, enabled: Optional[bool] = None,
                        telegram_alerts: Optional[bool] = None,
                        include_bos_alerts: Optional[bool] = None,
                        min_quote_volume_usd: Optional[float] = None,
                        top_n: Optional[int] = None) -> Dict:
        """Update scanner settings. Returns the merged settings dict."""
        if enabled is not None:
            self._enabled = bool(enabled)
        if telegram_alerts is not None:
            self._telegram_alerts_enabled = bool(telegram_alerts)
        if include_bos_alerts is not None:
            self._include_bos_alerts = bool(include_bos_alerts)
        if min_quote_volume_usd is not None:
            try:
                v = float(min_quote_volume_usd)
                # Sanity bounds — prevent obviously wrong values
                if 0 <= v <= 1e12:
                    self._min_quote_volume_usd = v
            except (TypeError, ValueError):
                pass
        if top_n is not None:
            try:
                n = int(top_n)
                if 10 <= n <= 200:
                    self._top_n = n
            except (TypeError, ValueError):
                pass
        return self.get_settings()
    
    def get_settings(self) -> Dict:
        return {
            'enabled': self._enabled,
            'telegram_alerts': self._telegram_alerts_enabled,
            'include_bos_alerts': self._include_bos_alerts,
            'min_quote_volume_usd': self._min_quote_volume_usd,
            'top_n': self._top_n,
            'timeframe': SCAN_TIMEFRAME,
            'last_scan_at': self._last_scan_at.isoformat() if self._last_scan_at else None,
            'last_scan_summary': self._last_scan_summary,
            'is_scanning': self._is_scanning,
        }
    
    # ============================================================
    # Scheduler
    # ============================================================
    
    def start(self):
        """Start the background scheduler. Idempotent."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return  # Already running
        self._stop_event.clear()
        t = threading.Thread(target=self._scheduler_loop,
                             name='Top100OBScanner-scheduler',
                             daemon=True)
        t.start()
        self._scheduler_thread = t
        print('[TOP100-OB] Scheduler started')
    
    def stop(self):
        """Signal the scheduler to stop. Won't interrupt an in-progress scan."""
        self._stop_event.set()
        print('[TOP100-OB] Scheduler stop requested')
    
    def _scheduler_loop(self):
        """Wakes up every minute, runs scan if (a) enabled and (b) we're
        within the first ~2 minutes of a 4-hour boundary (00:01-00:02,
        04:01-04:02, etc.) AND we haven't scanned this boundary yet.
        
        Why minute-resolution polling instead of `schedule` lib: keeps it
        simple, no extra dep. The cost (60 wakeups/hour each doing a
        cheap timestamp check) is negligible.
        """
        last_scheduled_run_hour = -1
        while not self._stop_event.is_set():
            try:
                if self._enabled:
                    now = datetime.now(timezone.utc)
                    # 4H boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
                    is_4h_boundary = (now.hour % SCAN_INTERVAL_HOURS == 0)
                    is_in_window = (now.minute == 1 or now.minute == 2)
                    not_yet_run = (now.hour != last_scheduled_run_hour)
                    if is_4h_boundary and is_in_window and not_yet_run:
                        print(f'[TOP100-OB] Scheduled scan triggered at {now}')
                        self.scan(triggered_by='schedule')
                        last_scheduled_run_hour = now.hour
            except Exception as e:
                # Never let scheduler thread die — log and continue
                print(f'[TOP100-OB] Scheduler error: {e}')
            # Wait 60 seconds OR until stop signal. The wait() returns
            # True if stop was signalled; we exit on that.
            if self._stop_event.wait(timeout=60):
                break
        print('[TOP100-OB] Scheduler loop exited')
    
    # ============================================================
    # Scan execution
    # ============================================================
    
    def scan(self, triggered_by: str = 'manual') -> Dict:
        """Execute one full scan. Thread-safe — concurrent calls return
        immediately with a "busy" indicator rather than queuing.
        
        Returns dict with summary stats: {triggered_by, started_at,
        finished_at, symbols_scanned, ob_found, fresh_choch, fresh_bos,
        mitigated, errors}.
        """
        # Non-blocking lock acquire — if scan already running, bail out.
        # Caller can poll get_settings()['is_scanning'] to check.
        if not self._scan_lock.acquire(blocking=False):
            return {'status': 'busy', 'message': 'scan already in progress'}
        
        try:
            self._is_scanning = True
            return self._do_scan(triggered_by)
        finally:
            self._is_scanning = False
            self._scan_lock.release()
    
    def _do_scan(self, triggered_by: str) -> Dict:
        """Inner scan body — caller must hold the lock."""
        started_at = datetime.utcnow()
        print(f'[TOP100-OB] Starting scan (triggered_by={triggered_by})')
        
        # Fetch the universe
        try:
            universe = self.md.fetch_top_perp_symbols(
                n=self._top_n,
                min_quote_volume_usd=self._min_quote_volume_usd)
        except Exception as e:
            print(f'[TOP100-OB] fetch_top_perp_symbols failed: {e}')
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
        
        print(f'[TOP100-OB] Universe: {len(universe)} symbols')
        
        # Track aggregate counters for summary
        symbols_scanned = 0
        ob_found = 0
        fresh_choch_alerts: List[Dict] = []
        fresh_bos_alerts: List[Dict] = []
        mitigations: List[Dict] = []
        errors = 0
        
        for entry in universe:
            symbol = entry['symbol']
            try:
                result = self._scan_symbol(symbol, entry)
                symbols_scanned += 1
                if result['ob_present']:
                    ob_found += 1
                if result['event_type'] == 'created':
                    if result['created_by_tag'] == 'CHoCH':
                        fresh_choch_alerts.append(result)
                    elif result['created_by_tag'] == 'BOS':
                        fresh_bos_alerts.append(result)
                elif result['event_type'] == 'mitigated':
                    mitigations.append(result)
            except Exception as e:
                errors += 1
                print(f'[TOP100-OB] Error scanning {symbol}: {e}')
            # Throttle between API calls — kind to Binance, predictable load
            time.sleep(self._throttle_ms / 1000.0)
        
        # Daily-ish housekeeping: drop old history rows. Cheap query, fine
        # to run on every scan since it has a hard date filter.
        try:
            deleted = self.db.cleanup_top100_ob_history(retention_days=HISTORY_RETENTION_DAYS)
            if deleted > 0:
                print(f'[TOP100-OB] Cleaned up {deleted} old history rows')
        except Exception as e:
            print(f'[TOP100-OB] History cleanup error: {e}')
        
        finished_at = datetime.utcnow()
        duration_sec = (finished_at - started_at).total_seconds()
        
        summary = {
            'status': 'ok',
            'triggered_by': triggered_by,
            'started_at': started_at.isoformat(),
            'finished_at': finished_at.isoformat(),
            'duration_sec': round(duration_sec, 1),
            'symbols_scanned': symbols_scanned,
            'ob_found': ob_found,
            'fresh_choch_count': len(fresh_choch_alerts),
            'fresh_bos_count': len(fresh_bos_alerts),
            'mitigated_count': len(mitigations),
            'errors': errors,
        }
        self._last_scan_summary = summary
        self._last_scan_at = started_at
        
        print(f'[TOP100-OB] Scan complete: {summary}')
        
        # Send Telegram alert if anything fresh emerged
        try:
            self._maybe_send_telegram(fresh_choch_alerts, fresh_bos_alerts)
        except Exception as e:
            print(f'[TOP100-OB] Telegram alert error: {e}')
        
        return summary
    
    def _scan_symbol(self, symbol: str, ticker_entry: Dict) -> Dict:
        """Scan a single symbol. Compares with previous snapshot and writes
        snapshot + history rows. Returns dict describing what happened:
            {ob_present: bool, event_type: 'created'|'mitigated'|'replaced'|'unchanged'|'none',
             created_by_tag: str, bias: str, bar_high: float, bar_low: float,
             zone_str: str, age_hours: float, last_price: float,
             quote_volume_24h: float, symbol: str}
        """
        last_price = ticker_entry.get('last_price', 0.0)
        quote_vol = ticker_entry.get('quote_volume', 0.0)
        market_ctx = {
            'quote_volume_24h': quote_vol,
            'last_price': last_price,
            'price_change_24h': ticker_entry.get('price_change_pct', 0.0),
        }
        
        # Fetch klines on the configured timeframe
        klines = self.md.fetch_klines(symbol, limit=KLINES_LIMIT,
                                       interval=SCAN_TIMEFRAME)
        if not klines or len(klines) < 50:
            # Not enough data — clear any stale snapshot OB but keep
            # market_ctx tracking
            self.db.upsert_top100_ob_snapshot(symbol, market_ctx, None)
            return {'ob_present': False, 'event_type': 'none',
                    'symbol': symbol, 'created_by_tag': '',
                    'bias': '', 'bar_high': 0, 'bar_low': 0,
                    'zone_str': '', 'age_hours': 0,
                    'last_price': last_price, 'quote_volume_24h': quote_vol}
        
        # Drop the in-progress (still-forming) bar — Pine semantics
        # require fully-closed bars. Without this we'd get phantom OBs
        # that "appear" then "disappear" as the bar evolves.
        klines_closed = klines[:-1] if len(klines) > 1 else klines
        
        # Run structure detection then OB extraction (same as smc_scanner)
        try:
            structure = detect_smc_structure(
                klines_closed, internal_size=5, swing_size=50)
            ob = detect_last_order_block(
                klines_closed,
                structure['internal']['pivots'],
                structure['internal']['events'])
        except Exception as e:
            print(f'[TOP100-OB] {symbol} OB detection error: {e}')
            return {'ob_present': False, 'event_type': 'none',
                    'symbol': symbol, 'created_by_tag': '',
                    'bias': '', 'bar_high': 0, 'bar_low': 0,
                    'zone_str': '', 'age_hours': 0,
                    'last_price': last_price, 'quote_volume_24h': quote_vol}
        
        # === Compute Premium/Discount/Equilibrium zone for the OB ===
        # The zone classifies where the OB sits within the latest SWING
        # range (not internal — swing is what defines the macro trading
        # range). Standard SMC fib levels:
        #   pos_pct < 38.2  → Discount  (bottom third, buy-the-dip zone)
        #   pos_pct > 61.8  → Premium   (top third, sell-the-rally zone)
        #   else            → Equilibrium
        # zone_correct is True when:
        #   BULLISH OB in Discount (LONG entry from cheap zone — ideal)
        #   BEARISH OB in Premium  (SHORT entry from rich zone — ideal)
        # Anything else is "wrong zone" — mathematically valid OB but
        # poor R:R because price is not at an extreme yet.
        if ob and ob.get('bias'):
            zone = self._compute_zone(ob, structure.get('swing', {}).get('pivots', []))
            zone_correct = self._is_zone_correct(zone, ob['bias'])
            ob['zone'] = zone
            ob['zone_correct'] = zone_correct
        
        # Compare with previous snapshot to determine event type
        prev = self.db.get_top100_ob_snapshot(symbol)
        prev_has_ob = prev and prev.get('bias') is not None
        prev_created_at_t = prev.get('created_at_t') if prev else None
        
        event_type = 'unchanged'
        is_fresh = False
        history_payload = None
        
        if ob is not None and ob.get('bias'):
            # Currently have an OB
            if not prev_has_ob:
                # New OB where there was none before
                event_type = 'created'
                is_fresh = True
                history_payload = ob
            elif prev_created_at_t != ob.get('created_at_t'):
                # Different OB than before (created_at_t is the unique
                # identifier — same bar's OB will have same created_at_t).
                # Could be 'replaced' (same direction, newer) or just a
                # different OB after the prior one was mitigated.
                if prev.get('bias') == ob.get('bias'):
                    event_type = 'replaced'
                else:
                    event_type = 'created'  # Direction flip = brand new setup
                is_fresh = True
                history_payload = ob
            # else: same OB still valid → event_type stays 'unchanged'
            ob_present = True
        else:
            # Currently no OB
            if prev_has_ob:
                # Previous snapshot had an OB, now it's gone — mitigated
                event_type = 'mitigated'
                # History payload is the OB that just got mitigated, not
                # the (nonexistent) current one
                history_payload = {
                    'bias': prev.get('bias'),
                    'bar_high': prev.get('bar_high'),
                    'bar_low': prev.get('bar_low'),
                    'bar_time': prev.get('bar_time_ms'),
                    'created_by_tag': prev.get('created_by_tag'),
                }
            ob_present = False
        
        # Persist current state
        self.db.upsert_top100_ob_snapshot(symbol, market_ctx, ob, is_fresh)
        
        # Append history if a notable event occurred
        if event_type in ('created', 'replaced', 'mitigated'):
            self.db.add_top100_ob_history(
                symbol=symbol,
                event_type=event_type,
                ob_data=history_payload,
                price_at_event=last_price,
                quote_volume_24h=quote_vol)
        
        # Build result dict for caller
        bar_high = ob.get('bar_high', 0) if ob else 0
        bar_low = ob.get('bar_low', 0) if ob else 0
        zone_str = ''
        age_hours = 0.0
        if ob:
            zone_str = f'{self._fmt_price(bar_low)} — {self._fmt_price(bar_high)}'
            try:
                age_ms = int(time.time() * 1000) - int(ob.get('bar_time', 0))
                age_hours = max(0, age_ms / 1000 / 3600)
            except (TypeError, ValueError):
                pass
        
        return {
            'ob_present': ob_present,
            'event_type': event_type,
            'symbol': symbol,
            'created_by_tag': ob.get('created_by_tag', '') if ob else '',
            'bias': ob.get('bias', '') if ob else '',
            'bar_high': bar_high,
            'bar_low': bar_low,
            'zone_str': zone_str,
            'age_hours': age_hours,
            'last_price': last_price,
            'quote_volume_24h': quote_vol,
        }
    
    @staticmethod
    def _compute_zone(ob_data: Dict, swing_pivots: List[Dict]) -> Optional[str]:
        """Classify where an OB sits within the latest swing range.
        
        Uses standard SMC Fibonacci-based zones:
          pos_pct < 38.2   → 'Discount'    (bottom third)
          pos_pct > 61.8   → 'Premium'     (top third)
          otherwise        → 'Equilibrium'
        
        Returns None when the swing range can't be determined (not enough
        pivots, or invalid range). The UI treats None as "unknown zone"
        which is filtered out by the zone-correct filter — i.e. unknown
        is treated as not tradeable, conservative default.
        
        We use the LATEST swing high and LATEST swing low — they bound
        the most recent trading range. Older pivots are ignored even if
        they're more extreme, because we want the *current* range.
        """
        if not ob_data or 'bar_high' not in ob_data or 'bar_low' not in ob_data:
            return None
        
        latest_high = None  # latest swing HIGH price
        latest_low = None   # latest swing LOW price
        # Walk pivots in reverse chronological order. swing_pivots are
        # appended in order they're created, so reversed() gives newest
        # first. Pivot types: 'HH'/'LH' = high pivots, 'HL'/'LL' = low pivots.
        for p in reversed(swing_pivots):
            ptype = p.get('type', '')
            if ptype in ('HH', 'LH') and latest_high is None:
                latest_high = p.get('price')
            elif ptype in ('HL', 'LL') and latest_low is None:
                latest_low = p.get('price')
            if latest_high is not None and latest_low is not None:
                break
        
        if latest_high is None or latest_low is None:
            return None  # Not enough swing context yet
        if latest_high <= latest_low:
            return None  # Invalid (would yield zero or negative range)
        
        range_size = latest_high - latest_low
        # Use OB MIDPOINT for the zone test — using just bar_high or
        # bar_low could push borderline OBs into the "wrong" classification
        # depending on which boundary we pick. Midpoint is symmetric.
        ob_mid = (ob_data['bar_high'] + ob_data['bar_low']) / 2
        pos_pct = (ob_mid - latest_low) / range_size * 100
        
        if pos_pct < 38.2:
            return 'Discount'
        if pos_pct > 61.8:
            return 'Premium'
        return 'Equilibrium'
    
    @staticmethod
    def _is_zone_correct(zone: Optional[str], bias: Optional[str]) -> bool:
        """Returns True when the OB's zone aligns with its trade direction.
        
          BULLISH OB in Discount  → True   (LONG from cheap — ideal R:R)
          BEARISH OB in Premium   → True   (SHORT from rich — ideal R:R)
          Equilibrium / wrong-zone / unknown → False
        
        Used by the UI's "Correct Zone" filter to surface only setups where
        the OB direction agrees with where price is in the swing range.
        """
        if not zone or not bias:
            return False
        if bias == 'BULLISH':
            return zone == 'Discount'
        if bias == 'BEARISH':
            return zone == 'Premium'
        return False
    
    @staticmethod
    def _fmt_price(p: float) -> str:
        """Format a price for display, with reasonable precision based on scale."""
        if p == 0:
            return '0'
        if p >= 1000:
            return f'{p:,.0f}'
        if p >= 1:
            return f'{p:,.2f}'
        if p >= 0.01:
            return f'{p:.4f}'
        if p >= 0.0001:
            return f'{p:.6f}'
        return f'{p:.8f}'
    
    # ============================================================
    # Telegram
    # ============================================================
    
    def _maybe_send_telegram(self, fresh_choch: List[Dict],
                              fresh_bos: List[Dict]) -> None:
        """Send a batch alert if there are any fresh OBs and notifier is set.
        Silent if nothing fresh — no spam.
        """
        if not self._telegram_alerts_enabled:
            return
        if not self._notifier:
            return
        
        # Build the message — CHoCH always included, BOS only if user opted in
        items_to_alert = list(fresh_choch)
        if self._include_bos_alerts:
            items_to_alert.extend(fresh_bos)
        
        if not items_to_alert:
            return  # Nothing fresh worth alerting about
        
        # Sort by 24h volume desc — most-traded coins first
        items_to_alert.sort(
            key=lambda x: x.get('quote_volume_24h', 0), reverse=True)
        
        header_count = len(items_to_alert)
        title = f'⚡ TOP-100 4H OB Radar — {header_count} fresh setup'
        if header_count != 1:
            title += 's'
        
        lines = [title, '']
        for item in items_to_alert[:15]:  # Cap message length
            icon = '🟢' if item['bias'] == 'BULLISH' else '🔴'
            side = 'LONG' if item['bias'] == 'BULLISH' else 'SHORT'
            tag = item['created_by_tag']
            vol_m = item['quote_volume_24h'] / 1_000_000
            
            age = item.get('age_hours', 0)
            if age < 1:
                age_str = f'{int(age*60)}m ago'
            elif age < 24:
                age_str = f'{int(age)}h ago'
            else:
                age_str = f'{int(age/24)}d ago'
            
            lines.append(f'{icon} {item["symbol"]} — {side} {tag}')
            lines.append(f'   Zone: {item["zone_str"]} ({age_str})')
            lines.append(f'   24h vol: ${vol_m:,.0f}M')
            lines.append('')
        
        if header_count > 15:
            lines.append(f'... and {header_count - 15} more (see UI for full list)')
            lines.append('')
        
        lines.append('ℹ️ Informational only. Verify on chart before trading.')
        
        msg = '\n'.join(lines)
        try:
            self._notifier.send_message(msg)
            print(f'[TOP100-OB] Telegram alert sent ({header_count} items)')
        except Exception as e:
            print(f'[TOP100-OB] Telegram send error: {e}')


# ============================================================
# Singleton
# ============================================================

_scanner_instance: Optional[Top100OBScanner] = None


def get_top100_ob_scanner(telegram_notifier=None) -> Top100OBScanner:
    """Lazy singleton. First call creates the scanner; subsequent calls
    return the same instance. Pass telegram_notifier on first call to
    enable alerts.
    """
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = Top100OBScanner(telegram_notifier=telegram_notifier)
    elif telegram_notifier is not None and _scanner_instance._notifier is None:
        # Allow late binding of notifier (Flask app may construct things
        # in a different order than scanner needs)
        _scanner_instance._notifier = telegram_notifier
    return _scanner_instance
