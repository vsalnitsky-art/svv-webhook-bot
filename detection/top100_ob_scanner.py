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
# === Defaults ===
# Numbers below are starting points; user can change at runtime via UI.
DEFAULT_TOP_N = 100
DEFAULT_MIN_QUOTE_VOLUME_USD = 100_000_000  # $100M default per spec
DEFAULT_THROTTLE_MS = 600
KLINES_LIMIT = 400          # ATR-200 needs at least 200 + buffer

# Timeframe — was 4H originally; changed to 1H per user request for faster
# feedback. Configurable via update_settings(timeframe=...).
DEFAULT_SCAN_TIMEFRAME = '1h'
ALLOWED_SCAN_TIMEFRAMES = ('15m', '30m', '1h', '2h', '4h')

# Schedule cadence (minutes between scheduled scans). 1H scans tighter than
# the old 4H+30min approach — 10-minute cadence catches new OBs within
# half a candle while still respecting Binance/Bybit rate limits with
# 100 symbols × ~600ms throttle per scan.
DEFAULT_SCAN_INTERVAL_MIN = 10

# "Fresh" OB window. With 4H bars, 12h ≈ 3 candles; with 1H bars the same
# notion needs to scale down. Default 3h on 1H ≈ 3 candles, mirroring the
# original 12h-on-4H ratio. User-customizable too if they want stricter
# (1h) or looser (6h) freshness.
DEFAULT_FRESH_WINDOW_HOURS = 3

# Zone-correctness thresholds (percent of latest swing range):
#   BULLISH OB valid (zone-correct) when pct ≤ long_max_pct  (default 20)
#   BEARISH OB valid (zone-correct) when pct ≥ short_min_pct (default 80)
# These are stricter than the old SMC Fib defaults (38.2/61.8) because the
# user wants only deep-extreme OBs flagged as ★. Still configurable.
DEFAULT_LONG_MAX_PCT = 20.0
DEFAULT_SHORT_MIN_PCT = 80.0

HISTORY_RETENTION_DAYS = 30


class Top100OBScanner:
    """Daemon-backed scanner for TOP-N OBs on a configurable timeframe."""
    
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
        # New: TF + scheduling + zone thresholds
        self._timeframe = DEFAULT_SCAN_TIMEFRAME
        self._scan_interval_min = DEFAULT_SCAN_INTERVAL_MIN
        self._fresh_window_hours = DEFAULT_FRESH_WINDOW_HOURS
        self._long_max_pct = DEFAULT_LONG_MAX_PCT
        self._short_min_pct = DEFAULT_SHORT_MIN_PCT
        
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
                        top_n: Optional[int] = None,
                        timeframe: Optional[str] = None,
                        scan_interval_min: Optional[int] = None,
                        fresh_window_hours: Optional[float] = None,
                        long_max_pct: Optional[float] = None,
                        short_min_pct: Optional[float] = None) -> Dict:
        """Update scanner settings. Returns the merged settings dict.
        
        SIDE EFFECT — TF change clears the snapshot table:
        OBs computed on 4H aren't comparable to OBs on 1H (different
        bar boundaries, different swing pivots, different mitigation
        outcomes). Mixing them produces stale "fresh" markers and
        misleading age values. So when the user picks a new TF, we
        truncate the DB so the next scan starts clean.
        """
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
        if timeframe is not None:
            tf = str(timeframe).lower().strip()
            if tf in ALLOWED_SCAN_TIMEFRAMES and tf != self._timeframe:
                # TF actually changed — clear all stored OBs since they're
                # tied to the old TF's bar boundaries and no longer valid.
                # Logged so production has a paper trail of when this happens.
                old_tf = self._timeframe
                try:
                    cleared = self.db.clear_top100_ob_snapshots()
                    print(f'[TOP100-OB] TF changed {old_tf} → {tf}; '
                          f'cleared {cleared} snapshot rows for clean restart')
                except Exception as e:
                    print(f'[TOP100-OB] TF change cleanup failed: {e}')
                self._timeframe = tf
            elif tf in ALLOWED_SCAN_TIMEFRAMES:
                # Same TF re-asserted — no-op (don't truncate)
                self._timeframe = tf
        if scan_interval_min is not None:
            try:
                m = int(scan_interval_min)
                # Lower bound: 5 min (even faster invites rate-limit issues
                # on Binance for 100-symbol scans). Upper bound: 60 min
                # (longer than that and the scanner is essentially dormant).
                if 5 <= m <= 60:
                    self._scan_interval_min = m
            except (TypeError, ValueError):
                pass
        if fresh_window_hours is not None:
            try:
                h = float(fresh_window_hours)
                # 0.5h floor (anything tighter and few OBs ever qualify);
                # 24h ceiling (longer and "fresh" loses meaning).
                if 0.5 <= h <= 24:
                    self._fresh_window_hours = h
            except (TypeError, ValueError):
                pass
        if long_max_pct is not None:
            try:
                v = float(long_max_pct)
                if 0 <= v <= 100:
                    self._long_max_pct = v
            except (TypeError, ValueError):
                pass
        if short_min_pct is not None:
            try:
                v = float(short_min_pct)
                if 0 <= v <= 100:
                    self._short_min_pct = v
            except (TypeError, ValueError):
                pass
        # Sanity check: long_max should be ≤ short_min (otherwise the
        # filter would simultaneously block both directions in some range).
        # If user inverted them, restore defaults — silent correction
        # is friendlier than rejecting the save.
        if self._long_max_pct >= self._short_min_pct:
            self._long_max_pct = DEFAULT_LONG_MAX_PCT
            self._short_min_pct = DEFAULT_SHORT_MIN_PCT
        return self.get_settings()
    
    def get_settings(self) -> Dict:
        return {
            'enabled': self._enabled,
            'telegram_alerts': self._telegram_alerts_enabled,
            'include_bos_alerts': self._include_bos_alerts,
            'min_quote_volume_usd': self._min_quote_volume_usd,
            'top_n': self._top_n,
            'timeframe': self._timeframe,
            'scan_interval_min': self._scan_interval_min,
            'fresh_window_hours': self._fresh_window_hours,
            'long_max_pct': self._long_max_pct,
            'short_min_pct': self._short_min_pct,
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
        """Run scans every `scan_interval_min` minutes when enabled.
        
        Old behavior: locked to 4H bar boundaries (00:00, 04:00, ...) plus
        a 1-2 minute window. Made sense when the scanner was specifically
        4H-only — scans aligned to bar close.
        
        New behavior: simple minute-cadence. We track the last scan's
        wall-clock time and trigger a new one when enough minutes have
        elapsed. Doesn't try to align to bar boundaries (1H/15m/etc), so
        a scan on a 1H deployment may catch the 1H bar a couple minutes
        after close — close enough for OB detection given that the
        underlying detector requires CONFIRMED bars only (klines[:-1]).
        """
        last_scan_started_at: Optional[datetime] = None
        while not self._stop_event.is_set():
            try:
                if self._enabled:
                    now = datetime.now(timezone.utc)
                    interval = max(1, int(self._scan_interval_min))
                    is_due = (
                        last_scan_started_at is None
                        or (now - last_scan_started_at).total_seconds()
                            >= interval * 60
                    )
                    if is_due:
                        last_scan_started_at = now
                        print(f'[TOP100-OB] Scheduled scan triggered at {now} '
                              f'(every {interval} min, TF={self._timeframe})')
                        self.scan(triggered_by='schedule')
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
        
        # Fetch klines on the configured timeframe (self._timeframe is
        # the user-selected TF; defaults to 1H but can be 15m/30m/2h/4h).
        klines = self.md.fetch_klines(symbol, limit=KLINES_LIMIT,
                                       interval=self._timeframe)
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
        # === Compute Discount/Mid/Premium zone for the OB ===
        # User-configurable thresholds (default 20% / 80%):
        #   pct ≤ long_max_pct (20)  → Discount  (BULLISH OB → zone-correct)
        #   pct ≥ short_min_pct (80) → Premium   (BEARISH OB → zone-correct)
        #   else                     → Mid       (never zone-correct)
        # zone_pct is also stored for the UI to display the exact percent —
        # users want to see "12%" or "85%" alongside the zone label.
        if ob and ob.get('bias'):
            pct = self._compute_zone_pct(ob, structure.get('swing', {}).get('pivots', []))
            zone, zone_correct = self._classify_zone(pct, ob['bias'])
            ob['zone'] = zone
            ob['zone_correct'] = zone_correct
            ob['zone_pct'] = pct  # raw percent for table display
        
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
    def _compute_zone_pct(ob_data: Dict, swing_pivots: List[Dict]
                           ) -> Optional[float]:
        """Compute where the OB midpoint sits within the latest swing
        range, returned as a percent (0-100).
        
        Range bounds = latest swing high (`HH` or `LH`) and latest swing
        low (`HL` or `LL`). We use the LATEST pivots — older ones are
        ignored even if they're more extreme — because the goal is to
        classify against the *current* trading range, not historical.
        
        Returns:
          float — percent rounded to 1 decimal. Can exceed [0, 100] when
                the OB sits beyond the latest swing range (rare).
          None  — range can't be determined (no pivots, inverted range).
        
        We use the OB MIDPOINT for the position test — using just bar_high
        or bar_low could push borderline OBs into a wrong zone depending
        on which boundary we pick. Midpoint is symmetric.
        """
        if not ob_data or 'bar_high' not in ob_data or 'bar_low' not in ob_data:
            return None
        
        latest_high = None  # latest swing HIGH price
        latest_low = None   # latest swing LOW price
        for p in reversed(swing_pivots):
            ptype = p.get('type', '')
            if ptype in ('HH', 'LH') and latest_high is None:
                latest_high = p.get('price')
            elif ptype in ('HL', 'LL') and latest_low is None:
                latest_low = p.get('price')
            if latest_high is not None and latest_low is not None:
                break
        
        if latest_high is None or latest_low is None:
            return None
        if latest_high <= latest_low:
            return None  # Invalid (would yield zero or negative range)
        
        range_size = latest_high - latest_low
        ob_mid = (ob_data['bar_high'] + ob_data['bar_low']) / 2
        pos_pct = (ob_mid - latest_low) / range_size * 100
        return round(pos_pct, 1)
    
    def _classify_zone(self, pct: Optional[float], bias: Optional[str]
                       ) -> tuple:
        """Threshold-based zone classification.
        
        Uses scanner's configured `long_max_pct` and `short_min_pct` to
        determine if an OB is in its "right" zone (high R:R extreme):
        
            BULLISH OB zone-correct  iff  pct ≤ long_max_pct   (deep low)
            BEARISH OB zone-correct  iff  pct ≥ short_min_pct  (deep high)
        
        Anything in between (or unknown) is NOT zone-correct.
        
        Returns (zone_label, zone_correct):
          zone_label: 'Discount' | 'Mid' | 'Premium' | None
                Display label for the table column. Discount = below
                long_max, Premium = above short_min, Mid = between, None
                = unknown range.
          zone_correct: bool
                True iff this OB's bias matches its zone (ideal R:R).
        
        Note: 'Discount' / 'Premium' here are RELATIVE TO USER THRESHOLDS,
        not the SMC Fib defaults (38.2/61.8). With the requested 20/80
        defaults, only the deepest 20% of either end gets labelled —
        much stricter than classic SMC.
        """
        if pct is None or bias is None:
            return None, False
        if pct <= self._long_max_pct:
            return 'Discount', (bias == 'BULLISH')
        if pct >= self._short_min_pct:
            return 'Premium', (bias == 'BEARISH')
        return 'Mid', False  # Between thresholds → not zone-correct
    
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
