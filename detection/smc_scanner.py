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
  'choch_or_bos'— Alert on EITHER: the fresh CHoCH itself, OR its later
                  BOS confirmation (union of the two modes above; never a
                  standalone BOS without a preceding CHoCH)

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

# Signal recency window (minutes). A CHoCH/BOS must be no older than this to
# fire an alert / forward to TM. Set to 0 = OFF (no age restriction — every
# new structure event can fire regardless of how old it is). Default OFF per
# user request: the 30-min hardcoded window was silently dropping valid BOS
# confirmations on slower pairs / higher timeframes where bars are 15m+.
DEFAULT_RECENCY_MINUTES = 0   # 0 = off (no recency filter)
ALLOWED_RECENCY_MINUTES = (0, 30, 60, 120)
KLINES_LIMIT = 3000           # bars to fetch per scan via paginated API.
                              # 3000 × 15m = ~31 days. Larger lookback gives
                              # the SMC structure detector enough history to
                              # stabilize trend state — matches what TV does
                              # by default. Tested against TV ZECUSDT.P: 1000
                              # bars produced a different CHoCH/BOS sequence
                              # because of insufficient warmup; 3000+ converges.
DEFAULT_TIMEFRAME = '5m'      # default timeframe
DEFAULT_INTERNAL_SIZE = 5     # default Pine Internal Structure size
MAX_WATCHLIST = 100

# Allowed timeframes (Binance format)
ALLOWED_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '4h']
ALLOWED_HTF_METHODS = ['EMA Cross', 'EMA Trend', 'Swing Structure', 'Internal Structure']

# Display labels (uppercase) for UI / Telegram
TIMEFRAME_LABELS = {
    '1m': '1M', '3m': '3M', '5m': '5M', '15m': '15M',
    '30m': '30M', '1h': '1H', '4h': '4H',
}

DB_KEY_WATCHLIST = 'smc_watchlist'
DB_KEY_WATCHLIST_SOURCES = 'smc_watchlist_sources'
# Per-coin metadata for the Tickr auto-pipeline: {symbol: {ts, score}}.
# Only tickr-sourced coins carry meta (added_ts for the 24h TTL, plus the
# opportunity score at add-time, used for lowest-score eviction). Manual
# coins have no meta — they never expire.
DB_KEY_WATCHLIST_META = 'smc_watchlist_meta'
DB_KEY_TRADEABLE = 'smc_tradeable'   # subset of watchlist that's tradeable
DB_KEY_SETTINGS = 'smc_settings'
DB_KEY_STATE = 'smc_last_events'   # tracks last seen event per symbol
DB_KEY_SIGNALS_PREFIX = 'smc_signals_'   # per-symbol: smc_signals_BTCUSDT, etc.
DB_KEY_DEDUP_STATE = 'smc_dedup_state_v1'  # authoritative per-symbol dedup gate state
DB_KEY_TRENDS_STATE = 'smc_trends_state_v1'  # last-known trend dot per symbol (restart cache)
SIGNALS_PERSIST_LIMIT = 50         # max signals stored per symbol

DEFAULT_SETTINGS = {
    'enabled': True,
    'alert_mode': DEFAULT_ALERT_MODE,
    'recency_minutes': DEFAULT_RECENCY_MINUTES,   # 0=off, else 30/60/120
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
    
    # === OB FILTER (Pine SMC_PRO_BOT__47_ Order Blocks) ===
    # Independent of HTF Bias filter. When enabled, every signal is gated
    # against the last valid Order Block on `ob_timeframe`:
    #   LONG signal requires last_ob.bias == BULLISH
    #   SHORT signal requires last_ob.bias == BEARISH
    # Missing OB (no BOS/CHoCH yet, or all OBs mitigated) blocks signals
    # in BOTH directions — user explicitly chose hard-block semantics for
    # this filter, identical to how HTF Bias works when no clear bias.
    # The OB itself is computed per-symbol on every scan tick and cached
    # in DB (table sob_smc_ob_state) so chart panel and signal gate share
    # the same source of truth.
    'ob_filter_enabled': False,
    'ob_filter_timeframe': '1h',  # 15m / 30m / 1h / 4h
    
    # === PD Zone Filter (threshold-based) ===
    # Gates signals based on CURRENT PRICE position within the trailing
    # range. Two configurable thresholds:
    #   long_max_pct  — block LONG  if pos_pct >= this value (default 75%)
    #   short_min_pct — block SHORT if pos_pct <= this value (default 25%)
    #
    # Semantics:
    #   • LONG entries valid in lower portion of range, blocked when
    #     price is too high (≥75% by default — "no buying expensive")
    #   • SHORT entries valid in upper portion of range, blocked when
    #     price is too low (≤25% by default — "no selling cheap")
    #   • Middle zone (25% < pos_pct < 75%) — both LONG and SHORT
    #     allowed, no filter restriction
    #
    # Range is computed Pine-faithfully using `trailing.top`/`bottom`
    # (Pine SMC_PRO_BOT__47_ lines 835-841): the running max/min of
    # high/low from the latest swing pivot to the current bar. Pine
    # resets these to the pivot price when a new swing forms, then
    # maintains running max/min on every subsequent bar.
    #
    # Default ON — matches standard SMC sentiment "buy low, sell high".
    # Default TF is 1H — wider context produces more stable, meaningful
    # zones than a 15m range that compresses every few hours.
    'use_pd_zone_filter': True,
    'pd_zone_timeframe': '1h',     # 15m / 30m / 1h / 4h
    # Timeframe for CTR/STC (Cyclic Trend Reversal). GENERAL setting — drives
    # CTR everywhere (chart badge, WATCHLIST, FF queues), since all consumers
    # read the one forecast-engine cache. Default 1H.
    'ctr_timeframe': '1h',         # 5m / 15m / 30m / 1h / 2h / 4h
    'pd_long_max_pct': 75.0,       # block LONG  if pos_pct >= this (0-100)
    'pd_short_min_pct': 25.0,      # block SHORT if pos_pct <= this (0-100)
    
    # === Forecast Filter (Pine 1H/4H multi-horizon prediction) ===
    # Independent of OB / PD / HTF filters. Reads `forecast_1h` and
    # `forecast_4h` from the ForecastEngine cache (computed by 6 Fibonacci
    # horizons on the respective TF — trend + momentum + volatility scoring).
    #
    # Gate behaviour:
    #   - Filter enabled flag controls per-TF whether that forecast acts as
    #     a gate. When OFF, the forecast is ignored entirely.
    #   - `side` from each forecast: +1 = LONG, -1 = SHORT, 0 = neutral.
    #   - `side == 0` always blocks when the filter is ON ("no clear
    #     direction = no entry") regardless of combine mode.
    #   - When both filters ON, `forecast_combine_mode` decides logic:
    #       AND — signal passes only if BOTH 1H and 4H sides match
    #       OR  — signal passes if at LEAST ONE matches AND the other is
    #             not actively contradicting (side != opposite)
    #
    # If the forecast cache is empty (engine hasn't computed yet, or fetch
    # error) → block. Same "err on the side of blocking" semantics as the
    # other filters.
    'forecast_1h_filter_enabled': False,
    'forecast_4h_filter_enabled': False,
    'forecast_combine_mode': 'AND',  # 'AND' or 'OR' — only used when BOTH enabled
    
    # === Volumized Order Blocks (port of TradingView "Volumized OBs" indicator) ===
    # Informational trend detector: finds the latest formed OB (Bull/Bear)
    # on the configured timeframe, and exposes its direction as a "trend"
    # signal in the UI panel and watchlist. Does NOT gate signals — works
    # alongside the existing CHoCH+BOS alert mode.
    #
    # All 5 algorithm-affecting Pine inputs are exposed verbatim:
    #   volumized_timeframe        — TF (15m/30m/1h/4h)
    #   volumized_swing_length     — Pine `swingLength`, default 10 (min 3)
    #   volumized_ob_end_method    — Pine `obEndMethod`: 'Wick' or 'Close'
    #   volumized_max_atr_mult     — Pine `maxATRMult`, default 3.5 (OB size filter)
    #   volumized_zone_count       — Pine `zoneCount`: 'One'/'Low'/'Medium'/'High'
    #   volumized_combine_obs      — Pine `combineOBs`, merge overlapping zones
    #
    # Default ON — gives user immediate visual feedback on trend direction.
    # Default TF is 1H — same rationale as PD Zone (stable structure).
    'use_volumized_ob': True,
    'volumized_timeframe': '1h',
    'volumized_swing_length': 10,
    'volumized_ob_end_method': 'Wick',     # 'Wick' or 'Close'
    'volumized_max_atr_mult': 3.5,
    'volumized_zone_count': 'Low',          # 'One' / 'Low' / 'Medium' / 'High'
    'volumized_combine_obs': True,
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
        # High-water mark: newest crossover bar time (to_t, ms) we've ever
        # PROCESSED per symbol. The seen-set above is capped at 200 ids, so on
        # symbols with >200 lifetime events the oldest ids get evicted and would
        # re-enter "new events" on a later scan — making weeks-old structure
        # look new and (with recency OFF) fire at a stale level. This guard
        # ensures we never fire an event whose bar isn't strictly newer than
        # anything we've already acted on, independent of the recency setting.
        # Rebuilt on each first scan (incl. after a restart), so no persistence
        # needed: first scan never fires, it only records + sets the mark.
        self._event_hwm: Dict[str, int] = {}  # {symbol: newest_to_t_ms}
        
        # HTF Bias cache per symbol: {symbol: {'bias', 'method', ...}}
        self._htf_cache: Dict[str, Dict] = {}
        
        # PD Zone cache per symbol — Premium/Discount/Equilibrium classification
        # of CURRENT PRICE within the latest swing range. Updated on every
        # scan tick; consumed by the signal gate (when use_pd_zone_filter
        # is on) and the chart-header badge.
        # {symbol: {'zone': str|None, 'pct': float|None, 'updated_at': float}}
        self._pd_zone_cache: Dict[str, Dict] = {}
        # Volumized OB Trend cache — keyed by symbol. Each entry holds
        # {'trend': 'LONG'|'SHORT'|None, 'meta': {...}, 'tf': '1h',
        #  'updated_at': float}. Populated per-scan-tick, consumed by
        # /api/smc/state for watchlist trend column, and by chart_data
        # for the chart-header trend badge. Cleared per-tick via the
        # same eviction path as _pd_zone_cache when a symbol falls out
        # of the watchlist.
        self._volumized_trend_cache: Dict[str, Dict] = {}
        
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

        # Last-known trend dot per symbol — {symbol: 1|-1|0}. Persisted to DB
        # after every scan and reloaded at startup so the watchlist consensus
        # (and the colored circles) have an INSTANT value right after a restart
        # instead of all-flat until the first full scan of 40+ coins finishes
        # (which can take a while). get_state() falls back to this whenever a
        # symbol hasn't been freshly scanned yet this run.
        self._trends_persisted: Dict[str, int] = {}
        self._load_trends_state()
        
        self._scan_count = 0
        self._errors = 0
        
        self._settings = self._load_settings()
        self._watchlist = self._load_watchlist()
        self._tradeable = self._load_tradeable()
        # Load signals only for symbols currently in watchlist
        self._load_all_signals()
        # Restore authoritative dedup gate state. On first run (no blob
        # yet) seed it from markers so legacy installs behave sensibly,
        # then persist so it becomes the source of truth going forward.
        if not self._load_dedup_state():
            self._seed_dedup_from_markers()
            self._persist_dedup_state()
    
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

    def _get_sources(self) -> dict:
        """{symbol: 'manual'|'tickr'} — how each watchlist coin was added."""
        if not self.db:
            return {}
        try:
            s = self.db.get_setting(DB_KEY_WATCHLIST_SOURCES, {})
            return s if isinstance(s, dict) else {}
        except Exception:
            return {}

    def _set_symbol_source(self, symbol: str, source: str):
        src = self._get_sources()
        # 'tickr_ff' = the FF-optimized Tickr pick (user-managed, no auto-TTL/
        # eviction — those only touch plain 'tickr').
        src[symbol] = source if source in ('manual', 'tickr', 'tickr_ff') else 'manual'
        if self.db:
            try:
                self.db.set_setting(DB_KEY_WATCHLIST_SOURCES, src)
            except Exception as e:
                print(f"[SMC] Watchlist sources persist error: {e}")

    def _clear_symbol_source(self, symbol: str):
        src = self._get_sources()
        if symbol in src:
            del src[symbol]
            if self.db:
                try:
                    self.db.set_setting(DB_KEY_WATCHLIST_SOURCES, src)
                except Exception:
                    pass

    def get_watchlist_sources(self) -> dict:
        """Public: {symbol: source} for coins currently in the watchlist.
        Symbols with no recorded source default to 'manual' (legacy adds)."""
        with self._lock:
            src = self._get_sources()
            return {s: src.get(s, 'manual') for s in self._watchlist}

    def symbols_by_source(self, source: str) -> List[str]:
        """Watchlist symbols currently tagged with the given source."""
        with self._lock:
            src = self._get_sources()
            return [s for s in self._watchlist if src.get(s, 'manual') == source]

    def remove_by_source(self, source: str) -> list:
        """Remove every watchlist coin tagged with `source`. Returns removed
        symbols. Used to clear the whole 📡 Tickr-FF batch at once."""
        targets = self.symbols_by_source(source)
        removed = []
        for sym in targets:
            if self.remove_symbol(sym).get('ok'):
                removed.append(sym)
        if removed:
            print(f"[SMC] Removed {len(removed)} '{source}' coins: {removed}")
        return removed

    # ---- Tickr auto-pipeline metadata (added_ts + opportunity score) ----

    def _get_meta(self) -> dict:
        """{symbol: {'ts': float, 'score': float}} for tickr coins."""
        if not self.db:
            return {}
        try:
            m = self.db.get_setting(DB_KEY_WATCHLIST_META, {})
            return m if isinstance(m, dict) else {}
        except Exception:
            return {}

    def _set_meta(self, symbol: str, ts: float, score: float):
        m = self._get_meta()
        m[symbol] = {'ts': float(ts), 'score': float(score)}
        if self.db:
            try:
                self.db.set_setting(DB_KEY_WATCHLIST_META, m)
            except Exception as e:
                print(f"[SMC] Watchlist meta persist error: {e}")

    def _clear_meta(self, symbol: str):
        m = self._get_meta()
        if symbol in m:
            del m[symbol]
            if self.db:
                try:
                    self.db.set_setting(DB_KEY_WATCHLIST_META, m)
                except Exception:
                    pass

    def get_watchlist_meta(self) -> dict:
        """Public: meta for tickr coins currently in the watchlist.
        {symbol: {ts, score}}. Manual coins are absent (no TTL)."""
        with self._lock:
            m = self._get_meta()
            return {s: m[s] for s in self._watchlist if s in m}

    def touch_tickr_symbol(self, symbol: str, score: float) -> Dict:
        """A tickr coin already in the watchlist fired again → refresh its
        added_ts (extend the 24h TTL) and update its score. No-op for coins
        that aren't tickr-sourced (e.g. manual)."""
        symbol = self._normalize_symbol(symbol)
        with self._lock:
            if symbol not in self._watchlist:
                return {'ok': False, 'reason': 'not in watchlist'}
            src = self._get_sources().get(symbol, 'manual')
            if src != 'tickr':
                return {'ok': False, 'reason': 'not tickr-sourced'}
            self._set_meta(symbol, time.time(), score)
        return {'ok': True, 'symbol': symbol}

    def expire_tickr_symbols(self, ttl_secs: int) -> list:
        """Remove tickr coins whose added_ts is older than ttl_secs.
        Manual coins are never touched (no meta → skipped). Returns the
        list of removed symbols."""
        now = time.time()
        with self._lock:
            meta = self._get_meta()
            src = self._get_sources()
            expired = [
                sym for sym in list(self._watchlist)
                if src.get(sym) == 'tickr'
                and sym in meta
                and (now - float(meta[sym].get('ts', now))) >= ttl_secs
            ]
        removed = []
        for sym in expired:
            r = self.remove_symbol(sym)
            if r.get('ok'):
                removed.append(sym)
        if removed:
            print(f"[SMC] Tickr TTL expired → removed {removed}")
        return removed

    def evict_lowest_tickr(self, below_score: float) -> Optional[str]:
        """Make room near MAX_WATCHLIST by removing the tickr coin with the
        LOWEST stored score, but only if that score is below `below_score`
        (the incoming candidate). Manual coins are never evicted. Returns
        the evicted symbol, or None if nothing was evicted."""
        with self._lock:
            meta = self._get_meta()
            src = self._get_sources()
            tickr = [(sym, float(meta.get(sym, {}).get('score', 0)))
                     for sym in self._watchlist if src.get(sym) == 'tickr']
        if not tickr:
            return None
        tickr.sort(key=lambda t: t[1])  # lowest score first
        weakest_sym, weakest_score = tickr[0]
        if weakest_score >= below_score:
            return None  # incoming isn't stronger than our weakest → no evict
        r = self.remove_symbol(weakest_sym)
        if r.get('ok'):
            print(f"[SMC] Evicted weakest tickr {weakest_sym} "
                  f"(score {weakest_score}) for stronger candidate "
                  f"(score {below_score})")
            return weakest_sym
        return None
    
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
        """Load persisted signal markers for every watchlist symbol.
        
        NOTE: This NO LONGER seeds _last_signal_dir (the dedup gate state).
        Dedup state is now persisted independently (DB_KEY_DEDUP_STATE) and
        is authoritative — so a manual reset survives restarts and settings
        reloads instead of being clobbered by re-seeding from old markers.
        See _load_dedup_state / _seed_dedup_from_markers / _persist_dedup_state.
        """
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
                                # Legacy markers (pre-2026-06-22) have no
                                # status — they were only ever recorded for
                                # opened trades, so default to 'opened'.
                                'status': str(item.get('status', 'opened')),
                                'reason': str(item.get('reason', '')),
                                'paper': bool(item.get('paper', False)),
                            })
                    if cleaned:
                        self._signal_markers[symbol] = cleaned
                        loaded += len(cleaned)
            except Exception as e:
                print(f"[SMC] Signal load error for {symbol}: {e}")
        if loaded:
            print(f"[SMC] Loaded {loaded} persisted signal markers across "
                  f"{len(self._signal_markers)} symbols")
    
    def _seed_dedup_from_markers(self):
        """Seed _last_signal_dir from the most recent signal marker per
        symbol. Only used on FIRST run (when no authoritative persisted
        dedup state exists yet) so legacy installs get a sensible initial
        dedup gate. After that, _persist_dedup_state owns the truth.
        """
        for symbol, markers in self._signal_markers.items():
            if not markers:
                continue
            try:
                last = max(markers, key=lambda m: m['time'])
                self._last_signal_dir[symbol] = last['side']
            except Exception:
                pass
    
    def _persist_dedup_state(self):
        """Save the dedup gate state (_last_signal_dir) to DB as the
        authoritative source. Called after every mutation: signal fire,
        Vol-flip reset, manual reset, clear_signals. A symbol absent from
        the saved dict = OPEN gate (ready to fire / manually reset).
        """
        if not self.db:
            return
        try:
            self.db.set_setting(DB_KEY_DEDUP_STATE, dict(self._last_signal_dir))
        except Exception as e:
            print(f"[SMC] dedup state persist error: {e}")
    
    def _load_dedup_state(self) -> bool:
        """Restore _last_signal_dir from the persisted authoritative blob.
        Returns True if a blob existed (so we should NOT re-seed from
        markers), False if this is a first run.
        
        Symbols present with 'LONG'/'SHORT' → gate locked that direction.
        Symbols absent → gate OPEN (includes manually-reset symbols, whose
        absence is exactly what we want to preserve across restarts).
        """
        if not self.db:
            return False
        stored = self.db.get_setting(DB_KEY_DEDUP_STATE, None)
        if isinstance(stored, dict):
            self._last_signal_dir = {
                k: v for k, v in stored.items() if v in ('LONG', 'SHORT')
            }
            print(f"[SMC] Restored authoritative dedup state: "
                  f"{len(self._last_signal_dir)} symbols locked")
            return True
        return False
    
    def _persist_trends_state(self, trends: Dict[str, int]):
        """Save the latest trend-dot snapshot to DB so it survives restarts.
        Called at the end of each scan cycle. Only non-flat (±1) entries are
        worth keeping — a flat (0) reads the same as 'no data' on reload, so
        dropping them keeps the blob small.
        """
        if not self.db:
            return
        try:
            snapshot = {k: int(v) for k, v in trends.items() if v in (1, -1)}
            self._trends_persisted = snapshot
            self.db.set_setting(DB_KEY_TRENDS_STATE, snapshot)
        except Exception as e:
            print(f"[SMC] trends state persist error: {e}")

    def _load_trends_state(self):
        """Restore the last-known trend dots from DB at startup, so the
        watchlist consensus has an immediate (stale-but-useful) value before
        the first full scan completes. Overwritten per-symbol as fresh scans
        land.
        """
        if not self.db:
            return
        try:
            stored = self.db.get_setting(DB_KEY_TRENDS_STATE, None)
            if isinstance(stored, dict):
                self._trends_persisted = {
                    k: int(v) for k, v in stored.items() if int(v) in (1, -1)
                }
                print(f"[SMC] Restored {len(self._trends_persisted)} cached "
                      f"trend dots from last run")
        except Exception as e:
            print(f"[SMC] trends state load error: {e}")

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

    def _record_marker(self, symbol: str, event: Dict, side: str,
                       status: str, reason: str = '',
                       is_paper: bool = False, entry_price: float = 0.0):
        """Record a chart marker for a signal with an explicit status.

        status:
          'opened'   — a real or paper position was actually opened (bright
                       green LONG / red SHORT dot on the chart)
          'rejected' — the signal was blocked by a filter (OB / PD Zone /
                       Forecast / Quality Gate / tradeable list). Shown as a
                       muted grey marker; clicking it reveals `reason`.

        We deliberately do NOT record 'duplicate' (same-direction) signals —
        those are not new trades and only add chart noise.

        Markers are deduped on (time, side, status) so a re-scan of the same
        bar doesn't stack identical markers.
        """
        to_t = event.get('to_t', 0) or 0
        t_sec = int(to_t // 1000) if to_t > 1e12 else int(to_t)
        persisted = False
        with self._lock:
            markers = self._signal_markers.setdefault(symbol, [])
            if not any(m['time'] == t_sec and m.get('side') == side
                       and m.get('status', 'opened') == status
                       for m in markers):
                markers.append({
                    'time': t_sec,
                    'price': float(entry_price or 0),
                    'side': side,
                    'status': status,
                    'reason': reason,
                    'paper': bool(is_paper),
                })
                if len(markers) > SIGNALS_PERSIST_LIMIT:
                    self._signal_markers[symbol] = markers[-SIGNALS_PERSIST_LIMIT:]
                persisted = True
        if persisted:
            self._persist_signals(symbol)
    
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
                self._persist_dedup_state()
                return {'ok': True, 'cleared': sym}
            else:
                cleared = list(self._signal_markers.keys())
                self._signal_markers.clear()
                self._last_signal_dir.clear()
                for s in cleared:
                    self._delete_signals(s)
                self._persist_dedup_state()
                return {'ok': True, 'cleared': cleared, 'count': len(cleared)}
    
    def reset_dedup(self, symbol: Optional[str] = None) -> Dict:
        """Manually reset the dedup gate (force OPEN) and PERSIST it.
        
        After this, the symbol's gate is open and — crucially — stays open
        across restarts and settings reloads because the dedup state is now
        authoritative in DB. The symbol then waits for the NEXT new
        algorithm action: _seen_events prevents re-firing the same event in
        the current session, and after a restart the first scan only records
        existing events (no alert). A genuinely new CHoCH/BOS (or a Vol flip)
        is what re-engages the gate.
        """
        with self._lock:
            if symbol:
                sym = self._normalize_symbol(symbol)
                self._last_signal_dir.pop(sym, None)
                self._persist_dedup_state()
                return {'ok': True, 'reset': sym}
            else:
                self._last_signal_dir.clear()
                self._persist_dedup_state()
                return {'ok': True, 'reset': 'all'}
    
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
    
    def add_symbol(self, symbol: str, source: str = 'manual',
                   score: float = 0.0) -> Dict:
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
            self._set_symbol_source(symbol, source)
            # Tickr coins carry meta (added_ts for TTL + score for eviction).
            # Manual coins get none — they never expire.
            if source == 'tickr':
                self._set_meta(symbol, time.time(), score)
            self._persist_watchlist()
            return {'ok': True, 'symbol': symbol, 'source': source,
                    'watchlist': list(self._watchlist)}
    
    def remove_symbol(self, symbol: str) -> Dict:
        symbol = self._normalize_symbol(symbol)
        with self._lock:
            if symbol in self._watchlist:
                self._watchlist.remove(symbol)
                self._clear_symbol_source(symbol)
                self._clear_meta(symbol)
                self._persist_watchlist()
                # Clean up state
                self._pending_choch.pop(symbol, None)
                self._first_scan_done.pop(symbol, None)
                self._seen_events.pop(symbol, None)
                self._htf_cache.pop(symbol, None)
                self._pd_zone_cache.pop(symbol, None)
                self._volumized_trend_cache.pop(symbol, None)
                self._signal_markers.pop(symbol, None)
                self._last_signal_dir.pop(symbol, None)
                self._persist_dedup_state()
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
                # Clear cached SMC OB state — consistent with user's spec
                # that OB info lives only as long as the symbol is watched.
                try:
                    from storage.db_operations import get_db
                    get_db().delete_smc_ob_state_for_symbol(symbol)
                except Exception as e:
                    print(f"[SMC] OB cleanup error for {symbol}: {e}")
                
                # === Volumized Radar hook ===
                # If the symbol was being tracked by the radar (24h TTL),
                # a manual remove counts as a "user rejection" — bump the
                # manual counter and set cooldown so the radar doesn't
                # immediately re-add the same symbol next scan. Best-effort:
                # if the DB call fails, the watchlist removal still completes.
                try:
                    from storage.db_operations import get_db as _gd
                    _gd().volradar_remove(symbol, reason='manual',
                                          cooldown_hours=6)
                except Exception as e:
                    print(f"[SMC] volradar manual hook error: {e}")
                
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
            allowed = ['enabled', 'alert_mode', 'recency_minutes', 'interval_secs', 'telegram_alerts',
                       'swing_size', 'timeframe', 'internal_size',
                       'deduplicate_signals',
                       'htf_enabled', 'htf_timeframe', 'htf_method',
                       'htf_ema_fast', 'htf_ema_slow', 'htf_ema_trend',
                       'htf_internal_size',
                       # OB filter
                       'ob_filter_enabled', 'ob_filter_timeframe',
                       # PD Zone filter (threshold-based)
                       'use_pd_zone_filter', 'pd_zone_timeframe',
                       'pd_long_max_pct', 'pd_short_min_pct',
                       # CTR/STC timeframe (general — drives CTR everywhere)
                       'ctr_timeframe',
                       # Forecast filter (1H/4H multi-horizon prediction)
                       'forecast_1h_filter_enabled', 'forecast_4h_filter_enabled',
                       'forecast_combine_mode',
                       # Volumized OB Trend (Pine port)
                       'use_volumized_ob', 'volumized_timeframe',
                       'volumized_swing_length', 'volumized_ob_end_method',
                       'volumized_max_atr_mult', 'volumized_zone_count',
                       'volumized_combine_obs']
            
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
            if self._settings.get('alert_mode') not in ('choch', 'choch_bos', 'choch_or_bos'):
                self._settings['alert_mode'] = DEFAULT_ALERT_MODE
            
            # Validate recency_minutes (0=off, or 30/60/120)
            try:
                rm = int(self._settings.get('recency_minutes', DEFAULT_RECENCY_MINUTES))
                if rm not in ALLOWED_RECENCY_MINUTES:
                    # Snap unknown values: <=0 → off, else nearest allowed
                    rm = 0 if rm <= 0 else min(
                        (a for a in ALLOWED_RECENCY_MINUTES if a > 0),
                        key=lambda a: abs(a - rm))
                self._settings['recency_minutes'] = rm
            except (TypeError, ValueError):
                self._settings['recency_minutes'] = DEFAULT_RECENCY_MINUTES
            
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
            
            # === OB Filter validation ===
            # Boolean toggle — coerce in case the UI sent string/None
            self._settings['ob_filter_enabled'] = bool(
                self._settings.get('ob_filter_enabled', False))
            # Timeframe — only allow the four user-requested options.
            # Anything else (including 5m, 1m, 1d) silently snaps to 1h.
            ALLOWED_OB_TFS = ('15m', '30m', '1h', '4h')
            if self._settings.get('ob_filter_timeframe') not in ALLOWED_OB_TFS:
                self._settings['ob_filter_timeframe'] = '1h'
            
            # === PD Zone Filter validation ===
            # Boolean toggle. Default True (set in DEFAULT_SETTINGS), but
            # update_settings can omit it — coerce explicitly so flipping
            # from True to False through the UI sticks. Without this cast,
            # an empty string from a missing form field would resolve as
            # truthy through `or` chains downstream.
            self._settings['use_pd_zone_filter'] = bool(
                self._settings.get('use_pd_zone_filter', True))
            # PD Zone TF: same allowed list as OB Filter. Falls back to
            # 1h if the supplied value is unknown — 1h is the default
            # because it gives a wider, more stable swing context than
            # 15m which compresses too often for clean PD zones.
            ALLOWED_PD_TFS = ('15m', '30m', '1h', '4h')
            if self._settings.get('pd_zone_timeframe') not in ALLOWED_PD_TFS:
                self._settings['pd_zone_timeframe'] = '1h'

            # CTR/STC timeframe (general setting).
            if self._settings.get('ctr_timeframe') not in ('5m', '15m', '30m', '1h', '2h', '4h'):
                self._settings['ctr_timeframe'] = '1h'
            
            # PD Zone thresholds — clamp to [0, 100] and ensure short ≤ long.
            # If the UI sends nonsense values (negative, >100, or short>long
            # which would create a contradictory gate), snap back to safe
            # defaults (75/25). We don't error — silent correction prevents
            # the filter from breaking signal flow when settings are typed
            # incorrectly.
            try:
                lm = float(self._settings.get('pd_long_max_pct', 75))
                sm = float(self._settings.get('pd_short_min_pct', 25))
                # Clamp to valid percentage range
                lm = max(0.0, min(100.0, lm))
                sm = max(0.0, min(100.0, sm))
                # Ensure short threshold ≤ long threshold. If user sets
                # them swapped, restore defaults — both being equal is
                # technically valid but creates a single fence point with
                # no buffer, which is unusual; let the user reset manually.
                if sm > lm:
                    lm, sm = 75.0, 25.0
                self._settings['pd_long_max_pct'] = lm
                self._settings['pd_short_min_pct'] = sm
            except (ValueError, TypeError):
                self._settings['pd_long_max_pct'] = 75.0
                self._settings['pd_short_min_pct'] = 25.0
            
            # === Volumized OB Trend validation ===
            # All 5 algorithm-affecting Pine parameters + the TF selector.
            # Each falls back to its DEFAULT_SETTINGS value on bad input,
            # never accepting silently corrupting values.
            ALLOWED_VOL_TFS = ('15m', '30m', '1h', '4h')
            if self._settings.get('volumized_timeframe') not in ALLOWED_VOL_TFS:
                self._settings['volumized_timeframe'] = '1h'
            
            try:
                sl = int(self._settings.get('volumized_swing_length', 10))
                # Pine minval = 3 (line 24: `input.int(10, ..., minval=3)`)
                self._settings['volumized_swing_length'] = max(3, sl)
            except (ValueError, TypeError):
                self._settings['volumized_swing_length'] = 10
            
            if self._settings.get('volumized_ob_end_method') not in ('Wick', 'Close'):
                self._settings['volumized_ob_end_method'] = 'Wick'
            
            try:
                mam = float(self._settings.get('volumized_max_atr_mult', 3.5))
                # Pine has no explicit max but 50× ATR is already absurd.
                # Negative or zero would disable all OBs (size > 0 always).
                self._settings['volumized_max_atr_mult'] = max(0.1, min(50.0, mam))
            except (ValueError, TypeError):
                self._settings['volumized_max_atr_mult'] = 3.5
            
            if self._settings.get('volumized_zone_count') not in ('One', 'Low', 'Medium', 'High'):
                self._settings['volumized_zone_count'] = 'Low'
            
            self._settings['volumized_combine_obs'] = bool(
                self._settings.get('volumized_combine_obs', True))
            
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

        # CONTINUOUS LIQ-MAP REGISTRATION: register EVERY watchlist coin with
        # the liquidation-map daemon each scan cycle. The liq-map only tracks
        # BTC/ETH + on-demand symbols (30-min TTL). Without this, watchlist
        # coins that aren't FF-flagged or recently viewed in the UI have no
        # fuel data → "Liq-палива немає даних". Re-requesting every scan keeps
        # the whole watchlist inside the TTL so all coins always have data.
        try:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm:
                for sym in list(self._watchlist):
                    try:
                        lm.request_symbol(sym)
                    except Exception:
                        pass
        except Exception as e:
            if self._errors <= 5:
                print(f"[SMC] liqmap register error: {e}")

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
                # Best-effort, never blocks scan loop on error.
                # IMPORTANT: we pass klines_closed (last bar dropped) so STC
                # is computed only on confirmed candles — matches Pine
                # behavior where ta.crossover() on barclose only fires after
                # the bar is sealed. Using full klines would let an in-progress
                # wick fire transient CTR signals that retract before close.
                try:
                    from detection.forecast_engine import get_forecast_engine
                    fe = get_forecast_engine()
                    if fe:
                        fe.update(symbol, ltf_klines=klines_closed,
                                  ctr_tf=self._settings.get('ctr_timeframe', '1h'))
                except Exception as fe_err:
                    if self._errors <= 5:
                        print(f"[SMC] Forecast update error for {symbol}: {fe_err}")
                
                # === Compute & persist OBs on all active timeframes ===
                # Three timeframes can need data this tick:
                #   1. main_tf (always — used for chart display, opposite_exit
                #      default, etc.)
                #   2. ob_filter_tf (entry gate, computed by _update_smc_ob)
                #   3. opposite_exit_tf (TM's exit rule, configurable)
                #   4. pd_zone_tf (PD Zone Filter — configurable, default 1h)
                # We build a per-TF cache so each unique TF is fetched and
                # SMC-detected at most once. Caches are scoped to this
                # symbol on this tick — discarded after.
                main_tf = self._settings.get('timeframe', '15m')
                ob_filter_tf = self._settings.get('ob_filter_timeframe', '1h')
                pd_zone_tf = self._settings.get('pd_zone_timeframe', '1h')
                
                # Collect TFs the TM might need (informational — TM gates
                # internally, but we still want OB rows ready in DB).
                opposite_exit_tf = main_tf  # safe default
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm is not None:
                        # Always populate the configured exit TF, even when
                        # use_opposite_ob_exit is OFF — that way toggling it
                        # ON in the UI doesn't have to wait a full scan tick
                        # for the row to materialize.
                        opposite_exit_tf = tm._settings.get(
                            'opposite_ob_exit_timeframe', '15m')
                except Exception:
                    pass
                
                from detection.ob_detector import detect_last_order_block
                from detection.smc_structure import detect_smc_structure
                from storage.db_operations import get_db
                
                # Per-TF cache: avoids duplicate fetch+detect on the same
                # TF. Pre-populate main_tf with already-computed data.
                isize_v = self.get_internal_size()
                ssize_v = int(self._settings.get('swing_size', 50))
                tf_data = {
                    main_tf: {
                        'klines_closed': klines_closed,
                        'structure': result_closed,
                    },
                }
                
                def _get_tf_data(tf):
                    """Lazy fetch+compute structure for a given TF, with
                    per-tick caching. Returns dict or None on failure.
                    
                    Limit raised from 700 → 2000 to match (and slightly
                    exceed) Pine's maxDistanceToLastBar=1750 from the
                    Volumized Order Blocks indicator. Pine processes the
                    last 1750 bars and accumulates state from up to 5000
                    (max_bars_back). With only 700 we'd see different OB
                    sequences than TradingView, especially on 4H where
                    700 bars = ~117 days but bear OBs from older swing
                    points could still be active. 2000 1H bars = ~83 days,
                    2000 4H = ~333 days — plenty of history without
                    runaway API cost (one fetch per scan tick, cached).
                    """
                    if tf in tf_data:
                        return tf_data[tf]
                    try:
                        if hasattr(md, 'fetch_klines') and \
                           'interval' in md.fetch_klines.__code__.co_varnames:
                            kl = md.fetch_klines(symbol, limit=2000, interval=tf)
                        else:
                            kl = md.fetch_klines(symbol, limit=2000)
                        if not kl or len(kl) < 220:
                            tf_data[tf] = None
                            return None
                        kl_closed = kl[:-1] if len(kl) >= 2 else kl
                        result = detect_smc_structure(
                            kl_closed,
                            internal_size=isize_v, swing_size=ssize_v)
                        tf_data[tf] = {
                            'klines_closed': kl_closed,
                            'structure': result,
                        }
                        return tf_data[tf]
                    except Exception as e:
                        if self._errors <= 5:
                            print(f"[SMC] TF data fetch error {symbol}@{tf}: {e}")
                        tf_data[tf] = None
                        return None
                
                # 1) main_tf — always compute using already-fetched klines
                try:
                    ob_main = detect_last_order_block(
                        klines=klines_closed,
                        pivots=result_closed.get('internal', {}).get('pivots', []),
                        events=result_closed.get('internal', {}).get('events', []),
                    )
                    get_db().upsert_smc_ob_state(symbol, main_tf, ob_main)
                except Exception as ob_main_err:
                    if self._errors <= 5:
                        print(f"[SMC] Main-TF OB compute error for {symbol}: {ob_main_err}")
                    ob_main = None
                
                # 2) ob_filter_tf — handled by _update_smc_ob below
                try:
                    self._update_smc_ob(symbol, md)
                except Exception as ob_err:
                    if self._errors <= 5:
                        print(f"[SMC] OB update error for {symbol}: {ob_err}")
                
                # 3) opposite_exit_tf — only compute OB if it's a NEW TF (not
                # main_tf and not ob_filter_tf). The cache helper handles the
                # lazy fetch; reuses prior compute if pd_zone_tf landed first.
                if opposite_exit_tf and \
                   opposite_exit_tf not in (main_tf, ob_filter_tf):
                    try:
                        td_extra = _get_tf_data(opposite_exit_tf)
                        if td_extra:
                            ob_extra = detect_last_order_block(
                                klines=td_extra['klines_closed'],
                                pivots=td_extra['structure'].get('internal', {}).get('pivots', []),
                                events=td_extra['structure'].get('internal', {}).get('events', []),
                            )
                            get_db().upsert_smc_ob_state(symbol, opposite_exit_tf, ob_extra)
                    except Exception as extra_err:
                        if self._errors <= 5:
                            print(f"[SMC] Extra-TF OB ({opposite_exit_tf}) error for {symbol}: {extra_err}")
                
                # === Compute PD Zone (Premium/Discount/Equilibrium) ===
                # PD Zone classifies the CURRENT PRICE within the trailing
                # range on the user-configured `pd_zone_timeframe` (default
                # 1H — wider context produces more stable zones than 15m).
                # _get_tf_data caches per-tick: if pd_zone_tf collides with
                # main_tf / ob_filter_tf / opposite_exit_tf we reuse the
                # already-fetched klines and structure. Otherwise one
                # additional API request fetches the right TF.
                # Cache stores percent for use by:
                #   1. Signal gate _pd_zone_filter_allows (when toggle on)
                #   2. chart_data badge in UI (with thresholds for color-coding)
                try:
                    pd_data = _get_tf_data(pd_zone_tf)
                    if pd_data:
                        pd_klines = pd_data['klines_closed']
                        pd_swing_pivots = pd_data['structure'].get(
                            'swing', {}).get('pivots', [])
                        pd_price = pd_klines[-1].get('p') if pd_klines else None
                        pd_pct = self._compute_pd_pct(
                            pd_klines, pd_swing_pivots, pd_price)
                    else:
                        pd_pct = None
                    with self._lock:
                        self._pd_zone_cache[symbol] = {
                            'pct': pd_pct,
                            'tf': pd_zone_tf,
                            'updated_at': time.time(),
                        }
                except Exception as pd_err:
                    if self._errors <= 5:
                        print(f"[SMC] PD Zone compute error for {symbol}: {pd_err}")
                
                # === Compute Volumized OB Trend ===
                # Pine port — finds the LATEST formed OB (Bull or Bear) and
                # exposes its direction as `trend` (LONG/SHORT). Purely
                # informational — does NOT gate CHoCH/BOS signals. Shown
                # in the chart-header trend badge and the watchlist
                # "Trend" column so the user knows which direction the
                # last institutional impulse pointed before deciding
                # whether to trust the next CHoCH/BOS alert.
                #
                # We pass FULL klines (incl. the forming bar) — same
                # rationale as PD Zone Filter: trailing extremes must
                # see the live price's reach or the OB filter's ATR
                # filter compares against stale ATR. The detector is
                # idempotent on the unchanged closed bars; the only
                # difference is the last entry now reflects the forming
                # bar instead of being missing.
                if self._settings.get('use_volumized_ob', True):
                    try:
                        vol_tf = self._settings.get('volumized_timeframe', '1h')
                        vol_data = _get_tf_data(vol_tf)
                        if vol_data:
                            from detection.volumized_ob import get_latest_ob_trend
                            vol_klines = vol_data.get('klines') or vol_data.get('klines_closed') or []
                            vol_result = get_latest_ob_trend(
                                vol_klines,
                                swing_length=int(self._settings.get('volumized_swing_length', 10)),
                                ob_end_method=self._settings.get('volumized_ob_end_method', 'Wick'),
                                max_atr_mult=float(self._settings.get('volumized_max_atr_mult', 3.5)),
                                zone_count=self._settings.get('volumized_zone_count', 'Low'),
                                combine_obs=bool(self._settings.get('volumized_combine_obs', True)),
                            )
                            new_vol_trend = vol_result.get('trend')  # 'LONG'|'SHORT'|None
                            with self._lock:
                                old_vol_trend = (
                                    self._volumized_trend_cache.get(symbol, {})
                                    .get('trend')
                                )
                                self._volumized_trend_cache[symbol] = {
                                    'trend': new_vol_trend,
                                    'meta': vol_result.get('trend_meta', {}),
                                    'tf': vol_tf,
                                    'updated_at': time.time(),
                                }
                            
                            # === Auto-reset dedup on Vol direction flip ===
                            # User-requested behavior: dedup state should
                            # always be aligned with the current Volumized
                            # direction. When Vol flips (LONG ↔ SHORT) we
                            # reset _last_signal_dir for the symbol so the
                            # next signal in the new direction is allowed
                            # to fire (gate goes OPEN). Without this, a
                            # symbol can get stuck "waiting for SHORT"
                            # even after Vol has flipped to SHORT and the
                            # previous LONG alert is no longer aligned with
                            # the current trend phase.
                            #
                            # Conditions: only reset on REAL direction flips
                            # (LONG → SHORT or SHORT → LONG). Transitions
                            # involving None (warmup or rejected by ATR) do
                            # NOT trigger a reset, since "direction
                            # unknown" isn't a true flip.
                            if (old_vol_trend in ('LONG', 'SHORT')
                                    and new_vol_trend in ('LONG', 'SHORT')
                                    and old_vol_trend != new_vol_trend):
                                with self._lock:
                                    prev = self._last_signal_dir.pop(symbol, None)
                                if prev is not None:
                                    self._persist_dedup_state()
                                    print(f"[SMC] {symbol} Vol flipped "
                                          f"{old_vol_trend}→{new_vol_trend}, "
                                          f"dedup reset (was locked={prev})")
                        else:
                            # No TF data — clear cache so stale entries
                            # don't linger (e.g., user changed TF and
                            # the new TF has no klines yet).
                            with self._lock:
                                self._volumized_trend_cache.pop(symbol, None)
                    except Exception as vol_err:
                        if self._errors <= 5:
                            print(f"[SMC] Volumized OB error for {symbol}: {vol_err}")
                
                # === Notify TM that OBs may have changed ===
                # TM reads its own configured TF from DB internally —
                # we just signal "hey, refresh your view".
                # Always fire (regardless of use_opposite_ob_exit toggle);
                # TM's gate keeps the toggle logic in one place.
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm:
                        tm.on_main_ob_update(symbol)
                except Exception as tm_err:
                    if self._errors <= 5:
                        print(f"[SMC] TM main-OB hook error for {symbol}: {tm_err}")
                
                # Alerts run on CLOSED bars only — won't fire from intra-bar
                # wicks that retract before close
                self._process_alerts(symbol, result_closed)
                
                # 200ms between symbols to spread load
                time.sleep(0.2)
            except Exception as e:
                if self._errors <= 10:
                    print(f"[SMC] {symbol} scan error: {e}")
                self._errors += 1
        
        # Snapshot the freshly-computed trend dots to DB so they survive a
        # restart and the watchlist consensus has an instant value next run.
        try:
            with self._lock:
                fresh_trends = self._build_trends(use_restart_fallback=False)
            self._persist_trends_state(fresh_trends)
        except Exception as e:
            if self._errors <= 5:
                print(f"[SMC] trends snapshot error: {e}")

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
    # SMC Order Block — per-tick update
    # ========================================
    
    def _update_smc_ob(self, symbol: str, md):
        """Compute the last valid SMC OB for this symbol on the OB-filter
        timeframe and persist it to DB.
        
        Runs on every scan tick for every watchlist symbol. The DB row is
        the single source of truth shared between:
          - chart panel (reads via get_chart_data → DB)
          - signal gate (reads in _process_alerts → DB)
        
        Best-effort: any error is logged but doesn't abort the surrounding
        scan loop. We always upsert (even with `None` ob_data when the
        detector found nothing) so the gate can distinguish "scanner ran,
        no OB exists" from "scanner never ran on this symbol".
        """
        ob_tf = self._settings.get('ob_filter_timeframe', '1h')
        # Whether or not the filter is enabled, we still maintain the OB
        # state so the chart panel always shows accurate info. Disabling
        # the filter just makes the gate skip the check.
        try:
            from detection.ob_detector import detect_last_order_block
            from storage.db_operations import get_db
        except Exception:
            return
        
        # Fetch klines for OB detection. We pull 700 bars to give ATR(200)
        # a 500-bar warmup (instead of just 200) — this stabilizes the
        # parsedHigh/Low volatility-filter classifications which are
        # sensitive to ATR jitter near the seed bar. Also gives a longer
        # pivot history so internal_size=5 has 700/5=140 potential pivots
        # to work with on the rolling window.
        OB_KLINES_LIMIT = 700
        try:
            if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames:
                klines = md.fetch_klines(symbol, limit=OB_KLINES_LIMIT, interval=ob_tf)
            else:
                klines = md.fetch_klines(symbol, limit=OB_KLINES_LIMIT)
        except Exception as e:
            # Network blip or rate-limit — don't crash, just skip this tick.
            return
        
        # Insufficient history check: ATR(200) Wilder needs at least 200 bars
        # for the SMA seed plus a few more for stability. Below 220 the
        # parsedHigh/Low classifications are unreliable.
        if not klines or len(klines) < 220:
            # Insufficient history for stable ATR. Leave any prior DB row
            # intact (don't wipe it), so the gate can still use the older
            # snapshot until we accumulate enough bars.
            return
        
        # Drop in-progress bar — Pine OB detection runs on closed bars only,
        # otherwise parsedHigh/Low and mitigation flips on intrabar wicks
        klines_closed = klines[:-1] if len(klines) >= 2 else klines
        
        # We need pivots and events on the same TF as the klines. Run the
        # SMC structure detector on this TF specifically — it's a separate
        # analysis from the main scan TF (e.g. main=15m, OB=1h).
        from detection.smc_structure import detect_smc_structure
        isize = self.get_internal_size()
        ssize = int(self._settings.get('swing_size', 50))
        result = detect_smc_structure(klines_closed,
                                       internal_size=isize,
                                       swing_size=ssize)
        internal = result.get('internal', {})
        
        ob = detect_last_order_block(
            klines=klines_closed,
            pivots=internal.get('pivots', []),
            events=internal.get('events', []),
        )
        
        # Persist (None ob means "computed but no valid OB" — explicit clear)
        try:
            db = get_db()
            db.upsert_smc_ob_state(symbol, ob_tf, ob)
        except Exception as e:
            if self._errors <= 5:
                print(f"[SMC] DB upsert OB error for {symbol}@{ob_tf}: {e}")
    
    def _ob_filter_allows(self, symbol: str, side: str) -> bool:
        """OB Filter gate decision for a fresh signal.
        
        Returns True only when ALL of these hold:
          1) DB has a computed row for this (symbol, ob_timeframe)
          2) Row has a non-null bias matching the signal side
          3) The OB was created by a CHoCH event (not BOS)
        
        Condition (3) is the strict-mode addition: Pine fires
        storeOrdeBlock on BOTH BOS and CHoCH events. CHoCH-created OBs
        mark fresh trend reversals — the entry pivot of a new trend.
        BOS-created OBs are continuation OBs, fired when an already-going
        trend pushes through another swing point. Both CHoCH-created and
        BOS-created OBs are valid filter sources — the gate cares only that
        an OB exists at this TF with a matching direction. The signal
        itself (CHoCH or BOS on the main TF) is what triggers entry; the
        OB filter just confirms we're not entering against an established
        institutional zone on the higher TF.
        
        Note that since obs[] is ordered newest-first in the detector and
        any new event (BOS or CHoCH) creates a fresh OB on top, checking
        only the latest OB is sufficient.
        
        We err on the side of blocking: any uncertainty = block.
        """
        try:
            from storage.db_operations import get_db
        except Exception:
            return False
        
        ob_tf = self._settings.get('ob_filter_timeframe', '1h')
        try:
            row = get_db().get_smc_ob_state(symbol, ob_tf)
        except Exception:
            return False
        
        if row is None:
            # Never computed — block until next scan tick produces a row.
            return False
        
        bias = row.get('bias')
        if bias is None:
            # Computed, but no valid OB exists at this timeframe right now.
            return False
        
        # Direction match — the only thing that matters now.
        # CHoCH-vs-BOS distinction was removed: both are valid OB sources,
        # the main-TF signal (CHoCH or BOS in candles) does the triggering,
        # the OB filter just confirms HTF direction is aligned.
        if side == 'LONG' and bias != 'BULLISH':
            return False
        if side == 'SHORT' and bias != 'BEARISH':
            return False
        
        return True
    
    @staticmethod
    def _compute_pd_pct(klines: List[Dict], swing_pivots: List[Dict],
                         current_price: float) -> Optional[float]:
        """Compute CURRENT PRICE position within the trailing range as a
        single percentage (0-100% in-range, can exceed for overshoots).
        
        Range definition is Pine-faithful (Pine SMC_PRO_BOT__47_ lines
        835-841 — `updateTrailingExtremes`). `trailing.top` and
        `trailing.bottom` are running max/min of high/low, RESET to the
        swing pivot's price when a new swing forms, then EXTEND via
        `max(high, trailing.top)` and `min(low, trailing.bottom)` on
        every subsequent bar.
        Practical effect: if price has crept above the latest swing
        high (without yet forming a new pivot), trailing.top tracks
        that new max — Pine's range expands to include it. Plain
        "latest swing pivot" would lag behind.
        
        Returns:
          float (rounded to 1 decimal) — position percent. Values can
                exceed [0, 100] when price has overshot the trailing
                extremes between bar updates (rare).
          None  — range can't be determined (no pivots yet, or
                  inverted/zero range).
        
        Zone classification was removed — caller (signal gate) compares
        the raw pct against configurable thresholds (pd_long_max_pct /
        pd_short_min_pct) for the actual filter decision.
        """
        if not swing_pivots or current_price is None or current_price <= 0:
            return None
        
        # Find latest swing high pivot AND latest swing low pivot.
        latest_high_pivot = None
        latest_low_pivot = None
        for p in reversed(swing_pivots):
            ptype = p.get('type', '')
            if ptype in ('HH', 'LH') and latest_high_pivot is None:
                latest_high_pivot = p
            elif ptype in ('HL', 'LL') and latest_low_pivot is None:
                latest_low_pivot = p
            if latest_high_pivot is not None and latest_low_pivot is not None:
                break
        
        if latest_high_pivot is None or latest_low_pivot is None:
            return None
        
        # === Compute trailing extremes (Pine updateTrailingExtremes) ===
        high_pivot_idx = latest_high_pivot.get('idx', 0) or 0
        low_pivot_idx = latest_low_pivot.get('idx', 0) or 0
        
        trailing_top = float(latest_high_pivot.get('price', 0))
        trailing_bottom = float(latest_low_pivot.get('price', 0))
        
        n = len(klines) if klines else 0
        if n > 0:
            for i in range(min(high_pivot_idx, n), n):
                h = klines[i].get('h', klines[i].get('p', 0))
                if h > trailing_top:
                    trailing_top = h
            for i in range(min(low_pivot_idx, n), n):
                l = klines[i].get('l', klines[i].get('p', 0))
                if l < trailing_bottom:
                    trailing_bottom = l
        
        if trailing_top <= trailing_bottom:
            # Inverted or zero range — happens at start of series before
            # pivots stabilize. Skip gracefully.
            return None
        
        range_size = trailing_top - trailing_bottom
        pos_pct = (current_price - trailing_bottom) / range_size * 100
        return round(pos_pct, 1)
    
    def _pd_zone_filter_allows(self, symbol: str, side: str) -> bool:
        """PD Zone gate — threshold-based.
        
        Two configurable thresholds:
          pd_long_max_pct  (default 75) — block LONG  if pos_pct >= this
          pd_short_min_pct (default 25) — block SHORT if pos_pct <= this
        
        Returns True (allow) when:
          • Filter is disabled
          • Pct unknown (defensive) — actually blocks instead, see below
          • For LONG:  pos_pct < pd_long_max_pct
          • For SHORT: pos_pct > pd_short_min_pct
        
        Returns False (block) when:
          • Pct hasn't been computed yet (insufficient pivots, fresh symbol)
          • Pct is at/beyond the side's threshold
        
        Note on "no pct → block": some users might prefer "no filter
        info → allow", but blocking on missing data is safer — SMC
        signals are rare enough that one extra scan tick of patience
        won't matter, and we avoid opening risky positions when the
        filter doesn't yet have data.
        """
        if not self._settings.get('use_pd_zone_filter', True):
            return True  # Filter disabled — pass through
        
        cached = self._pd_zone_cache.get(symbol)
        if not cached:
            print(f"[SMC] 🚫 PD Zone Filter blocked {symbol} {side}: "
                  f"pct not yet computed")
            return False
        
        pct = cached.get('pct')
        if pct is None:
            print(f"[SMC] 🚫 PD Zone Filter blocked {symbol} {side}: "
                  f"insufficient swing pivots (range undefined)")
            return False
        
        long_max = float(self._settings.get('pd_long_max_pct', 75.0))
        short_min = float(self._settings.get('pd_short_min_pct', 25.0))
        
        if side == 'LONG' and pct >= long_max:
            print(f"[SMC] 🚫 PD Zone Filter blocked {symbol} LONG: "
                  f"price at {pct}% — too high (threshold ≥{long_max}%)")
            return False
        if side == 'SHORT' and pct <= short_min:
            print(f"[SMC] 🚫 PD Zone Filter blocked {symbol} SHORT: "
                  f"price at {pct}% — too low (threshold ≤{short_min}%)")
            return False

        return True

    def get_pd_pct(self, symbol: str):
        """Cached Premium/Discount position % (0=bottom/Discount .. 100=top/
        Premium) for `symbol` on pd_zone_timeframe, or None if not computed.
        Public accessor for other modules (e.g. FF Queue-2 confluence)."""
        c = (self._pd_zone_cache.get((symbol or '').upper())
             or self._pd_zone_cache.get(symbol))
        return c.get('pct') if c else None

    def get_pd_thresholds(self):
        """(premium_min, discount_max) — pos_pct ≥ premium_min = Premium;
        pos_pct ≤ discount_max = Discount; else Equilibrium."""
        return (float(self._settings.get('pd_long_max_pct', 75.0)),
                float(self._settings.get('pd_short_min_pct', 25.0)))
    
    def _forecast_filter_allows(self, symbol: str, side: str) -> bool:
        """Forecast 1H / 4H gate.
        
        Settings consulted:
          forecast_1h_filter_enabled (bool)
          forecast_4h_filter_enabled (bool)
          forecast_combine_mode      ('AND' or 'OR')
        
        Forecast `side` values: +1 = LONG, -1 = SHORT, 0 = neutral.
        
        Decision matrix when only ONE filter is ON:
          - matches signal direction → allow
          - opposite                 → block
          - neutral (side == 0)      → block
          - cache miss / no data     → block (err on the side of caution,
            matches other filters' "no data → block" semantics)
        
        When BOTH filters are ON:
          AND — both 1H and 4H must individually allow the signal
          OR  — at least one must explicitly match the signal direction,
                AND the other must NOT be actively contradicting.
                Specifically, in OR mode:
                  - one filter allows + other allows  → pass
                  - one filter allows + other neutral → pass (no contradiction)
                  - one filter allows + other opposite → block
                  - both neutral                       → block
                  - any cache miss with the other not matching → block
        
        We block on cache miss because the ForecastEngine runs on its own
        cadence; if a symbol hasn't been processed yet, the safest answer
        is "wait". The next scan tick will revisit.
        """
        want_1h = bool(self._settings.get('forecast_1h_filter_enabled', False))
        want_4h = bool(self._settings.get('forecast_4h_filter_enabled', False))
        if not want_1h and not want_4h:
            return True  # No filter active
        
        # Map signal side label → forecast side value
        target = 1 if side == 'LONG' else -1
        
        # Pull cached forecast for this symbol
        forecast_1h = None
        forecast_4h = None
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe is not None:
                cached = fe.get(symbol)
                if cached:
                    forecast_1h = cached.get('forecast_1h')
                    forecast_4h = cached.get('forecast_4h')
        except Exception:
            pass
        
        def evaluate(fc, label):
            """Return one of: 'match', 'opposite', 'neutral', 'nodata'."""
            if not fc or not isinstance(fc, dict):
                return 'nodata'
            s = fc.get('side', None)
            if s is None:
                return 'nodata'
            if s == 0:
                return 'neutral'
            return 'match' if s == target else 'opposite'
        
        r1 = evaluate(forecast_1h, '1H') if want_1h else None
        r4 = evaluate(forecast_4h, '4H') if want_4h else None
        
        # Single-filter mode — simple match check
        if want_1h and not want_4h:
            if r1 == 'match':
                return True
            print(f"[SMC] 🚫 Forecast 1H blocked {symbol} {side}: {r1}")
            return False
        if want_4h and not want_1h:
            if r4 == 'match':
                return True
            print(f"[SMC] 🚫 Forecast 4H blocked {symbol} {side}: {r4}")
            return False
        
        # Both ON — combine
        mode = str(self._settings.get('forecast_combine_mode', 'AND')).upper()
        if mode not in ('AND', 'OR'):
            mode = 'AND'
        
        if mode == 'AND':
            if r1 == 'match' and r4 == 'match':
                return True
            print(f"[SMC] 🚫 Forecast AND blocked {symbol} {side}: "
                  f"1H={r1}, 4H={r4}")
            return False
        else:  # OR
            # At least one explicit match, the other must not contradict
            if r1 == 'match' and r4 != 'opposite' and r4 != 'nodata':
                return True
            if r4 == 'match' and r1 != 'opposite' and r1 != 'nodata':
                return True
            # Edge: one match + other nodata — still block (we treat nodata
            # as caution; if user wants more permissive, they can disable
            # the unused TF entirely)
            print(f"[SMC] 🚫 Forecast OR blocked {symbol} {side}: "
                  f"1H={r1}, 4H={r4}")
            return False
    
    # ========================================
    # Alerts logic
    # ========================================
    
    def _process_alerts(self, symbol: str, result: Dict):
        # Note: SMC scanner no longer sends Telegram messages directly.
        # Telegram delivery is owned by Trade Manager (real or test mode),
        # which has its own per-mode toggles. The pipeline below still runs
        # regardless of any legacy 'telegram_alerts' setting in DB, because
        # it does signal markers, dedup state, and TM hooks — all of which
        # are independent of Telegram.
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
        
        # Crossover bar time (ms) — when the structure event actually fired.
        # This is the right notion of "how recent" an event is (from_t is the
        # OLD pivot being broken; to_t is when close crossed it).
        def ev_to_t(e):
            t = e.get('to_t', 0) or 0
            return int(t)
        
        seen = self._seen_events.setdefault(symbol, set())
        is_first_scan = not self._first_scan_done.get(symbol, False)
        
        # Candidate NEW events (their stable IDs aren't in `seen`). NOTE: due to
        # the 200-id cap below, this list can also contain RE-SURFACED old events
        # whose ids were evicted — the high-water-mark guard further down filters
        # those out before anything fires.
        candidate_new = []
        for ev in events:
            eid = ev_id(ev)
            if eid not in seen:
                candidate_new.append((eid, ev))
                seen.add(eid)
        
        # Bound seen set to avoid unbounded growth
        if len(seen) > 200:
            recent_ids = {ev_id(e) for e in events[-200:]}
            self._seen_events[symbol] = recent_ids
        
        if is_first_scan:
            # First scan: just record existing events. NEVER seed pending_choch
            # from history, because the BOS that "confirms" it would also be
            # historical and is recorded as seen on this same scan.
            # Set the high-water mark to the newest event so future scans can
            # never re-fire any of this history (even after seen-set eviction).
            self._event_hwm[symbol] = max((ev_to_t(e) for e in events), default=0)
            self._first_scan_done[symbol] = True
            print(f"[SMC] {symbol}: first scan recorded {len(events)} historical events "
                  f"(no alerts, no pending CHoCH from history)")
            return
        
        # === High-water-mark guard (root fix for re-surfaced old events) ===
        # Keep only candidates whose crossover bar is strictly newer than the
        # newest bar we've already processed. A re-surfaced old event (evicted
        # from `seen` by the 200-cap) has an old to_t <= hwm and is dropped here
        # — it stays recorded in `seen` but never reaches the firing pipeline.
        # This makes correctness independent of the recency setting: even with
        # recency OFF, stale structure from days/weeks ago can no longer fire.
        hwm = self._event_hwm.get(symbol, 0)
        dropped_stale = 0
        new_events = []
        for eid, ev in candidate_new:
            if ev_to_t(ev) > hwm:
                new_events.append((eid, ev))
            else:
                dropped_stale += 1
        if dropped_stale:
            print(f"[SMC] {symbol}: dropped {dropped_stale} re-surfaced stale "
                  f"event(s) below high-water mark (seen-set eviction guard)")
        # Advance the mark to the newest event we're about to process.
        if new_events:
            self._event_hwm[symbol] = max(
                hwm, max(ev_to_t(ev) for _, ev in new_events))
        
        if not new_events:
            return  # nothing changed since last scan
        
        # === Recency guard ===
        # We have two different needs:
        #   - For DIRECT alerts (mode='choch' on CHoCH, or BOS confirming pending
        #     CHoCH in mode='choch_bos'): the event must be FRESH so we don't
        #     alert on something that happened hours ago.
        #   - For SEEDING pending state in choch_bos mode: a CHoCH that creates
        #     the "anchor" for a future BOS confirmation should NOT be filtered
        #     by age — it's just preparation, no alert is sent. BOS may arrive
        #     much later (sometimes hours), and as long as the BOS itself is
        #     fresh when it confirms, that's a valid setup.
        #
        # So: always seed pending_choch for CHoCH events in choch_bos mode.
        # Apply recency only when actually firing alerts.
        # Recency window — configurable. 0 = OFF (no age restriction): every
        # new structure event is treated as "recent" and can fire. Otherwise
        # the event's bar must be no older than recency_minutes.
        recency_min = int(self._settings.get('recency_minutes', DEFAULT_RECENCY_MINUTES))
        recency_off = recency_min <= 0
        recent_threshold_secs = recency_min * 60
        now_ms = int(time.time() * 1000)
        
        mode = self._settings.get('alert_mode', DEFAULT_ALERT_MODE)
        
        for _, ev in new_events:
            tag = ev.get('tag')
            to_t = ev.get('to_t', 0) or 0
            # to_t is in ms (kline open time)
            to_age_secs = (now_ms - to_t) / 1000 if to_t > 1e10 else (now_ms / 1000 - to_t)
            
            is_recent = recency_off or (to_age_secs <= recent_threshold_secs)
            
            # === Forward BOS events to Trade Manager (always, before mode logic) ===
            # TM uses these to:
            #   1) Update Health Score evaluator state (BOS counts) — ALWAYS,
            #      regardless of TM enabled / shadow mode.
            #   2) Trigger BOS-N partial closes — only when a real position
            #      exists and TM is enabled. The on_bos_event method itself
            #      enforces those preconditions; we don't gate at the
            #      scanner level so shadow positions can also receive
            #      counter updates for evaluator/UI consistency.
            # NOTE: We removed the prior `tm.is_enabled()` gate that lived
            # here. It blocked all BOS routing in test_mode (TM disabled
            # but shadow positions active) which prevented evaluator stats
            # and any future shadow-side partial logic from ever firing.
            if tag == 'BOS' and is_recent:
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm:
                        tm.on_bos_event(symbol=symbol, direction=ev['dir'],
                                          level=ev['level'], bar_t=to_t)
                except Exception as e:
                    print(f"[SMC] TM BOS hook error: {e}")
            
            # === Forward fresh CHoCH events to TM (regardless of TM enabled) ===
            # TM uses these for: Reverse SMC exit, Forecast 1H Confluence exit.
            # Both rules can run in shadow mode (TM disabled but test_mode on)
            # to send Telegram-only signals without opening real positions.
            if tag == 'CHoCH' and is_recent:
                try:
                    from detection.trade_manager import get_trade_manager
                    tm = get_trade_manager()
                    if tm:
                        tm.on_choch_event(symbol=symbol, direction=ev['dir'],
                                           level=ev['level'], bar_t=to_t)
                except Exception as e:
                    print(f"[SMC] TM CHoCH hook error: {e}")
                # Let Fuel Filter drop a Queue-2 coin on an OPPOSITE chart CHoCH
                # (tracks the chart directly, independent of the signal pipeline).
                try:
                    from detection.fuel_filter import get_fuel_filter
                    ff = get_fuel_filter()
                    if ff:
                        ff.queue2_on_choch(symbol, ev['dir'])
                except Exception as e:
                    print(f"[SMC] FF Q2 CHoCH hook error: {e}")

            if mode == 'choch':
                # CHoCH-only mode — alerts must be on FRESH CHoCH
                if tag == 'CHoCH' and is_recent:
                    if not self._htf_allows(symbol, ev['dir']):
                        print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by HTF filter")
                    elif not self._dedup_allows(symbol, ev['dir']):
                        # _dedup_allows blocks for two possible reasons —
                        # log which one fired so the user can debug "why
                        # didn't this alert?" without reading the code.
                        side_label = 'LONG' if ev['dir'] == 'bull' else 'SHORT'
                        vt = (self._volumized_trend_cache.get(symbol, {})
                              .get('trend'))
                        if (self._settings.get('use_volumized_ob', True)
                                and vt in ('LONG', 'SHORT')
                                and vt != side_label):
                            print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked: "
                                  f"goes against Volumized direction (Vol={vt}, signal={side_label})")
                        else:
                            print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by dedup (already fired this direction)")
                    else:
                        self._send_alert(symbol, ev, mode='choch')
            
            elif mode == 'choch_or_bos':
                # "Either" mode (2026-06-14): fire if EITHER condition holds —
                #   (a) a fresh CHoCH (the 'choch' rule), OR
                #   (b) that CHoCH later confirmed by a BOS (the 'choch_bos'
                #       rule).
                # This is the UNION of the two existing modes, NOT a
                # standalone-BOS trigger: a BOS with no prior CHoCH never
                # alerts here. In practice (a) already fires on the CHoCH;
                # (b) adds a second alert when the same move is confirmed.
                if tag == 'CHoCH':
                    # (a) alert on the fresh CHoCH itself
                    if is_recent:
                        if not self._htf_allows(symbol, ev['dir']):
                            print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by HTF filter")
                        elif not self._dedup_allows(symbol, ev['dir']):
                            print(f"[SMC] {symbol} CHoCH {ev['dir']} blocked by dedup")
                        else:
                            self._send_alert(symbol, ev, mode='choch')
                    # ...and seed pending so a later BOS can confirm (b)
                    prev = self._pending_choch.get(symbol)
                    if not (prev and prev.get('to_t', 0) > to_t):
                        self._pending_choch[symbol] = {
                            'from_t': ev['from_t'], 'to_t': ev['to_t'],
                            'level': ev['level'], 'dir': ev['dir'],
                            'choch_event': ev,
                        }
                elif tag == 'BOS' and is_recent:
                    # (b) only fires when it confirms a pending CHoCH —
                    # never on its own.
                    pending = self._pending_choch.get(symbol)
                    if pending and pending['dir'] == ev['dir']:
                        if ev.get('to_t', 0) > pending.get('to_t', 0):
                            if not self._htf_allows(symbol, ev['dir']):
                                print(f"[SMC] {symbol} CHoCH+BOS {ev['dir']} blocked by HTF filter")
                            elif not self._dedup_allows(symbol, ev['dir']):
                                print(f"[SMC] {symbol} CHoCH+BOS {ev['dir']} blocked by dedup")
                                self._pending_choch.pop(symbol, None)
                            else:
                                self._send_alert(symbol, ev, mode='choch_bos',
                                                  choch_event=pending['choch_event'])
                                self._pending_choch.pop(symbol, None)
                    elif pending and pending['dir'] != ev['dir']:
                        self._pending_choch.pop(symbol, None)

            elif mode == 'choch_bos':
                if tag == 'CHoCH':
                    # Always seed pending — even if old. The BOS confirmation
                    # will check freshness on its own when it arrives.
                    # An opposite-direction CHoCH naturally overwrites the
                    # previous pending state (last-CHoCH wins).
                    prev = self._pending_choch.get(symbol)
                    if prev and prev.get('to_t', 0) > to_t:
                        # An older event arriving out of order — keep newer pending
                        pass
                    else:
                        self._pending_choch[symbol] = {
                            'from_t': ev['from_t'],
                            'to_t': ev['to_t'],
                            'level': ev['level'],
                            'dir': ev['dir'],
                            'choch_event': ev,
                        }
                elif tag == 'BOS' and is_recent:
                    # BOS must be fresh to fire alert (confirms recent breakout)
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
        
        Two-layer gate:
        
        Layer 1 — Volumized direction filter (when use_volumized_ob is True
        AND Vol direction is determined for this symbol). Blocks signals
        that go AGAINST current Vol trend. This is what makes the dedup
        pip in the UI consistent with the Vol pip — there is no possible
        state where Vol says SHORT and dedup is locked LONG, because no
        LONG signal could have fired against a SHORT Vol direction in
        the first place. The exception is when Vol is None (warmup or
        rejected by ATR) — we let signals through since direction is
        undetermined.
        
        Layer 2 — Pine `proDeduplicateInput` behavior (when
        deduplicate_signals is True). Blocks signals matching the last
        fired direction. After a Vol flip, _last_signal_dir is reset
        (in the scan loop), so the next aligned signal can fire.
        
        Returns False (blocked) if EITHER layer rejects. Returns True
        (allowed) if both layers pass or if both are disabled.
        """
        side_label = 'LONG' if event_dir == 'bull' else 'SHORT'
        
        # === Layer 1: Vol direction filter ===
        # Block signals that contradict the current Volumized trend.
        # Without this, a LONG signal could fire while Vol says SHORT,
        # locking dedup against the actual trend direction.
        if self._settings.get('use_volumized_ob', True):
            vol_trend = (self._volumized_trend_cache.get(symbol, {})
                         .get('trend'))
            if vol_trend in ('LONG', 'SHORT') and vol_trend != side_label:
                # Vol says one direction, signal says the other → block.
                # Don't touch _last_signal_dir; we never fired.
                return False
        
        # === Layer 2: Pine `proDeduplicateInput` ===
        if not self._settings.get('deduplicate_signals', True):
            return True
        last_side = self._last_signal_dir.get(symbol)
        if last_side is None:
            return True  # first signal ever (or after Vol-flip reset)
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
            side_label = 'LONG' if is_bull else 'SHORT'

            # 🧾 Activity log: record EVERY fresh qualified signal at the EARLIEST
            # point (before the scanner's OB/PD/Forecast filters), so a signal a
            # filter blocks below is never lost without a trace.
            try:
                from detection.activity_log import log_activity
            except Exception:
                log_activity = lambda *a, **k: None
            log_activity(symbol, 'signal', f'Свіжий сигнал {mode}', side=side_label, source='scanner')

            # === OB Filter gate ===
            # When the user has enabled OB Filter, we require directional
            # agreement between the signal and the LAST VALID OB on the
            # configured OB timeframe (read from DB — same source the chart
            # panel uses). Hard-block semantics: if OB is missing or
            # opposite, the signal is dropped completely (no markers, no
            # dedup state update, no TM hook). The user explicitly chose
            # this behavior.
            # Level for rejected-marker placement (entry computed later).
            evt_level = event.get('level', 0) or 0

            if self._settings.get('ob_filter_enabled', False):
                if not self._ob_filter_allows(symbol, side_label):
                    # Blocked — record a rejected marker so the user can see
                    # (and click for the reason) why no trade fired here.
                    print(f"[SMC] 🚫 OB Filter blocked {symbol} {side_label} signal")
                    _r = 'OB-фільтр заблокував (Order Block проти напрямку)'
                    self._record_marker(symbol, event, side_label,
                                        'rejected', _r, entry_price=evt_level)
                    log_activity(symbol, 'rejected', _r, side=side_label, source='scanner')
                    return
            
            # === PD Zone Filter (Premium/Discount) ===
            # Independent of OB Filter — this is a price-position filter,
            # OB is a structure filter. Both can be on simultaneously
            # for max selectivity. Same hard-block semantics as OB Filter:
            # blocked signals leave no marker and don't reach TM.
            #
            # The toggle defaults ON because trading against the zone
            # (e.g. LONG from Premium = high risk) is the most common
            # newbie mistake; default protection is more useful than
            # default permissiveness.
            if not self._pd_zone_filter_allows(symbol, side_label):
                _r = 'PD-зона заблокувала (вхід проти Premium/Discount)'
                self._record_marker(symbol, event, side_label,
                                    'rejected', _r, entry_price=evt_level)
                log_activity(symbol, 'rejected', _r, side=side_label, source='scanner')
                return  # Already logged inside the helper
            
            # === Forecast Filter (1H / 4H multi-horizon prediction) ===
            # Per-TF enable, combine mode (AND/OR) when both ON. Reads the
            # cached forecast computed by ForecastEngine on a separate
            # schedule (so this gate is cheap — DB cache lookup only).
            # Same hard-block semantics: blocked signal leaves no marker
            # and never reaches TM.
            if (self._settings.get('forecast_1h_filter_enabled', False)
                    or self._settings.get('forecast_4h_filter_enabled', False)):
                if not self._forecast_filter_allows(symbol, side_label):
                    _r = 'Forecast-фільтр заблокував (прогноз 1H/4H проти напрямку)'
                    self._record_marker(symbol, event, side_label,
                                        'rejected', _r, entry_price=evt_level)
                    log_activity(symbol, 'rejected', _r, side=side_label, source='scanner')
                    return  # Already logged inside the helper
            
            # Entry price = the structural break LEVEL of the event that fired
            # the signal: the BOS break level in CHoCH+BOS mode, or the CHoCH
            # break level in CHoCH mode. This is the exact price `close` crossed
            # to confirm the structure — i.e. the actual trigger. Pinning the
            # entry to this level keeps it sitting ON the structure drawn on the
            # chart, instead of drifting to wherever live price happens to be at
            # scan time. Live price is only a last-resort fallback if, for some
            # reason, the event carries no usable level.
            entry_price = event.get('level', 0)
            if not entry_price or entry_price <= 0:
                entry_price = self._get_live_price(symbol) or 0
            
            entry_str = self._fmt_price(entry_price)

            # Telegram notification is sent by Trade Manager — either via
            # _notify_open() for real positions (gated by tm.telegram_alerts)
            # or via _open_shadow() for test/paper mode (gated by
            # tm.test_telegram_alerts). The scanner itself does NOT send to
            # Telegram to avoid duplicate messages on a single signal.
            # When TM (real) AND test_mode are both off, no Telegram is sent —
            # this is the intended behavior (silent mode).

            # === Forward signal to Trade Manager FIRST ===
            # TM decides whether to actually open a position (real or shadow)
            # based on its filters (side gates, tradeable list, manual mode).
            # We only add a chart marker if TM confirms it opened something.
            trade_opened = False
            tm_status = 'rejected'
            tm_reason = 'Trade Manager unavailable'
            tm_is_paper = False
            try:
                from detection.trade_manager import get_trade_manager
                tm = get_trade_manager()
                if tm:
                    result = tm.on_signal(symbol=symbol, side=side_label,
                                          entry_price=entry_price, opened_by=mode)
                    # TM may return None (e.g. signal ignored: manual mode,
                    # side-gate disabled, dedup) — normalize so the .get()
                    # calls below never crash on NoneType.
                    result = result or {}
                    # TM returns {'real_opened','shadow_opened','status','reason','is_paper'}
                    trade_opened = result.get('real_opened') or result.get('shadow_opened')
                    tm_status = result.get('status',
                                           'opened' if trade_opened else 'rejected')
                    tm_reason = result.get('reason', '')
                    tm_is_paper = result.get('is_paper', False)
            except Exception as e:
                print(f"[SMC] TM hook error: {e}")
                tm_reason = f'TM error: {e}'

            # Record a chart marker based on the resolved status:
            #   opened    → bright LONG/SHORT dot (a new position was opened)
            #   rejected  → muted grey marker carrying the block reason
            #   duplicate → no marker (not a new trade; just chart noise)
            if tm_status == 'opened':
                self._record_marker(symbol, event, side_label, 'opened',
                                    is_paper=tm_is_paper, entry_price=entry_price)
                tag = '📝 Paper' if tm_is_paper else '✅ Position'
                print(f"[SMC] {tag} opened: {symbol} {side_label} @ {entry_str}")
            elif tm_status == 'duplicate':
                print(f"[SMC] ↺ Duplicate {symbol} {side_label} "
                      f"(already in position) — no marker")
            else:
                self._record_marker(symbol, event, side_label, 'rejected',
                                    tm_reason, entry_price=entry_price)
                print(f"[SMC] ⊘ Signal rejected: {symbol} {side_label} "
                      f"@ {entry_str} — {tm_reason}")

            # Update last-direction state (used by dedup gate). Pine updates
            # this even when dedup is OFF, so toggling dedup ON later doesn't
            # cause a sudden re-fire of the prior direction.
            self._last_signal_dir[symbol] = side_label
            self._persist_dedup_state()
            
            # === Volumized OB Radar hook ===
            # If this symbol was added by the radar (within the 24h TTL),
            # the SMC signal firing means it "graduated" out of radar tracking.
            # The DB call deletes the metadata row and bumps times_signal_fired.
            # The symbol stays in the watchlist as a normal item — same as
            # any other after a signal. Lazy import to avoid circular import.
            try:
                graduated = self.db.volradar_mark_signal_fired(symbol)
                if graduated:
                    print(f"[SMC] {symbol} graduated from Volumized Radar tracking (signal fired)")
            except Exception as e:
                print(f"[SMC] volradar_mark_signal_fired error: {e}")
        except Exception as e:
            print(f"[SMC] Alert send error: {e}")
    
    def _get_cached_klines(self, symbol: str) -> Optional[List[Dict]]:
        """Return the cached LTF klines for a symbol (the TF the bot trades),
        or None if not scanned yet. Used by the Hold-Confidence analyzer so
        it reuses scan data instead of refetching."""
        with self._lock:
            cached = self._cache.get(symbol)
        if cached and cached.get('klines'):
            return cached['klines']
        return None

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
    
    def _build_volumized_chart_meta(self, symbol: str) -> Dict:
        """Enrich the cached Volumized trend meta for chart rendering.
        
        Cache stores `start_time` in ms (matching klines `t` field which
        is ms-epoch). LightweightCharts expects time values in SECONDS,
        so we compute `start_time_sec` here and add it alongside the
        original. The frontend uses `start_time_sec` for the OB box's
        x-axis position; `start_time` (ms) is kept for tooltip display
        and any other consumers that prefer ms.
        
        Returns the meta dict (possibly empty if no trend cached yet).
        Empty/no-trend case still returns {} so frontend code can
        gate on `Object.keys(meta).length > 0` instead of null-checking.
        """
        meta = dict(self._volumized_trend_cache.get(symbol, {}).get('meta', {}))
        if not meta:
            return {}
        # Convert start_time (ms) → start_time_sec (s) for the chart.
        # break_time same treatment if present (breaker OBs).
        st_ms = meta.get('start_time')
        if isinstance(st_ms, (int, float)) and st_ms > 0:
            # ms epochs are ~1.7e12 in 2024; sec epochs are ~1.7e9. If the
            # value is already < 1e11, assume seconds and don't divide.
            if st_ms > 1e11:
                meta['start_time_sec'] = int(st_ms / 1000)
            else:
                meta['start_time_sec'] = int(st_ms)
        bt_ms = meta.get('break_time')
        if isinstance(bt_ms, (int, float)) and bt_ms > 0:
            if bt_ms > 1e11:
                meta['break_time_sec'] = int(bt_ms / 1000)
            else:
                meta['break_time_sec'] = int(bt_ms)
        return meta
    
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
            # Normalize markers so the frontend always gets status/reason/paper,
            # even for legacy markers persisted before these fields existed.
            signals = [{
                'time': m.get('time'),
                'price': m.get('price', 0),
                'side': m.get('side'),
                'status': m.get('status', 'opened'),
                'reason': m.get('reason', ''),
                'paper': m.get('paper', False),
            } for m in self._signal_markers.get(symbol, [])]
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
        
        # === Forecast 1H + 4H + CTR (from forecast_engine cache) ===
        forecast_1h = None
        forecast_4h = None
        ctr = None
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe:
                cached_fc = fe.get(symbol)
                if cached_fc:
                    forecast_1h = cached_fc.get('forecast_1h')
                    forecast_4h = cached_fc.get('forecast_4h')
                    ctr = cached_fc.get('ctr')
        except Exception:
            pass
        
        # === Decision Center verdict ===
        # ONE unified analytical output that combines:
        #   - Entry Score for both LONG and SHORT
        #   - Probability distribution (softmax)
        #   - Recommended direction with verdict (good/marginal/poor)
        #   - One-line plain-English rationale
        #   - Live Health Score for any open position on this symbol
        #
        # The chart panel renders this as a single Decision block — no more
        # 3 separate badges (Entry-Long / Entry-Short / Health) crowding
        # the header. The block expands on click to show the full breakdown
        # for users who want to verify the bot's reasoning.
        decision = None
        try:
            from detection.trade_manager import get_trade_manager
            tm = get_trade_manager()
            if tm:
                last_price = None
                if ohlc:
                    try:
                        last_price = float(ohlc[-1].get('close') or 0)
                    except Exception:
                        last_price = None
                if last_price and last_price > 0:
                    decision = tm.compute_decision(symbol, last_price)
        except Exception as e:
            print(f"[SMC] chart_data decision error: {e}")
        
        # === Compute last valid Order Block (Pine SMC_PRO_BOT__47_) ===
        # === Last valid Order Block (Pine SMC_PRO_BOT__47_) ===
        # Read from DB cache (table sob_smc_ob_state) — the SAME source
        # the OB Filter signal gate consults. This guarantees the chart
        # panel and the gate never disagree about the current OB state.
        # The DB row is updated by the scan loop on every tick via
        # _update_smc_ob() at the user-configured ob_filter_timeframe.
        # If no row exists yet (symbol just added; scanner hasn't run),
        # fall back to inline compute on the chart's TF — better to show
        # something approximate than a blank badge.
        last_ob = None
        ob_tf_used = self._settings.get('ob_filter_timeframe', '1h')
        try:
            from storage.db_operations import get_db
            row = get_db().get_smc_ob_state(symbol, ob_tf_used)
            if row and row.get('bias'):
                # DB row has actual OB data
                last_ob = {
                    'bias': row['bias'],
                    'bar_high': row['bar_high'],
                    'bar_low': row['bar_low'],
                    'bar_time': row['bar_time'],
                    'bar_idx': row['bar_idx'],
                    'created_at_idx': row['created_at_idx'],
                    'created_at_t': row['created_at_t'],
                    'created_by_tag': row.get('created_by_tag', ''),
                }
            elif row is None:
                # Fallback — scanner hasn't run yet for this (symbol, TF).
                # Compute inline on the chart's main TF as a best-effort
                # display so user doesn't see an empty badge for a long
                # time after adding a symbol.
                try:
                    from detection.ob_detector import detect_last_order_block
                    klines_closed = klines[:-1] if len(klines) >= 2 else klines
                    last_ob = detect_last_order_block(
                        klines=klines_closed,
                        pivots=internal.get('pivots', []),
                        events=internal.get('events', []),
                    )
                except Exception as e:
                    print(f"[SMC] OB inline-fallback error for {symbol}: {e}")
            # row exists with bias=None → last_ob stays None (no valid OB)
        except Exception as e:
            print(f"[SMC] chart_data OB-read error for {symbol}: {e}")
        
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
            # Forecast 1H + 4H + CTR (Pine PRO indicators) — kept for legacy
            # consumers and the chart's auxiliary debug tooltip
            'forecast_1h': forecast_1h,
            'forecast_4h': forecast_4h,
            'ctr': ctr,
            # Unified Decision Center verdict — primary advisory output
            'decision': decision,
            # === Last valid Order Block (Pine SMC_PRO_BOT__47_) ===
            # The user wants ONLY the most recent unmitigated internal OB.
            # Computed via the exact Pine algorithm: parsedHigh/parsedLow
            # volatility filter (ATR(200) seed), storeOrdeBlock on every
            # internal BOS/CHoCH event, deleteOrderBlocks mitigation each bar.
            # We use INTERNAL pivots/events (size=5) since the Pine input
            # `internalOrderBlocksSizeInput` defaults to 5 and the user works
            # primarily on internal structure for entries.
            # Returns None when no OB has survived mitigation.
            'last_ob': last_ob,
            # OB Filter UI metadata — chart panel uses these to show the
            # source TF in the badge tooltip and the filter status (active
            # vs informational). Independent from the OB itself: even when
            # ob_filter_enabled=False we still publish last_ob for display.
            'ob_filter_enabled': bool(self._settings.get('ob_filter_enabled', False)),
            'ob_filter_timeframe': ob_tf_used,
            # === PD Zone (Premium/Discount/Equilibrium) badge data ===
            # Where current price sits within the latest swing range on
            # the Structure Detection TF. UI renders as a badge in the
            # chart-header next to OB / Forecast / CTR.
            # Filter toggle is published separately so the badge can show
            # a 🔒 prefix when the filter is actively gating signals.
            # === PD Zone — threshold-based filter data ===
            # No more zone classification — just the raw pct and the
            # configured thresholds. UI uses the thresholds for color-
            # coding the badge: green when within both bounds, red/yellow
            # when crossing either threshold.
            'pd_zone_pct': self._pd_zone_cache.get(symbol, {}).get('pct'),
            'pd_zone_filter_enabled': bool(self._settings.get('use_pd_zone_filter', True)),
            'pd_zone_timeframe': self._settings.get('pd_zone_timeframe', '1h'),
            'pd_long_max_pct': float(self._settings.get('pd_long_max_pct', 75.0)),
            'pd_short_min_pct': float(self._settings.get('pd_short_min_pct', 25.0)),
            # === Volumized OB Trend ===
            # Trend = direction of latest formed OB on volumized_timeframe.
            # 'LONG' = latest OB is Bull, 'SHORT' = latest OB is Bear,
            # None = no OB detected yet (warmup or filtered out by ATR).
            # trend_meta carries price levels + volume for the chart badge
            # to render e.g. "🟢 LONG • Bull OB $58k-$59k • vol 1.2M".
            # We also enrich meta with `start_time_sec` (seconds) so the
            # frontend doesn't need to know whether the raw start_time is
            # ms or seconds — LightweightCharts expects seconds.
            'volumized_trend': self._volumized_trend_cache.get(symbol, {}).get('trend'),
            'volumized_trend_meta': self._build_volumized_chart_meta(symbol),
            'volumized_timeframe': self._settings.get('volumized_timeframe', '1h'),
            'volumized_enabled': bool(self._settings.get('use_volumized_ob', True)),
            # Swing Structure (size=swing_size)
            'swing_pivots': fmt_pivots(swing),
            'swing_events': fmt_events(swing),
            'swing_trend': swing.get('trend', 0),
            'strong_high': strong_high,
            'weak_low': weak_low,
            'klines_count': len(ohlc),
            'updated_at': int(time.time()),
        }
    
    def _build_trends(self, use_restart_fallback: bool = True) -> Dict[str, int]:
        """Compute the per-symbol trend dot map ({symbol: 1|-1|0}).

        When the HTF filter is active and a symbol has a clear HTF bias, the
        dot reflects the HTF direction (matches the chart and the alerts).
        Otherwise it falls back to the internal-structure trend from the live
        scan cache.

        When use_restart_fallback is True (UI/state path), symbols that haven't
        been freshly scanned yet this run reuse the trend persisted from the
        previous run, so the watchlist consensus has an immediate value after a
        restart instead of all-flat. When False (persistence path), only freshly
        computed values are returned so we never re-persist stale data as fresh.

        Caller must hold self._lock.
        """
        htf_settings = self.get_htf_settings()
        htf_active = htf_settings.get('enabled', False)
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
            scanned = sym in self._cache
            if effective == 0:
                c = self._cache.get(sym)
                if c:
                    try:
                        effective = c.get('analysis', {}).get('internal', {}).get('trend', 0)
                    except:
                        effective = 0
            if effective == 0 and not scanned and use_restart_fallback:
                effective = self._trends_persisted.get(sym, 0)
            trends[sym] = effective
        return trends

    def get_state(self) -> Dict:
        with self._lock:
            htf_settings = self.get_htf_settings()
            htf_active = htf_settings.get('enabled', False)

            # Build per-symbol trend map (with restart fallback so the
            # watchlist consensus is populated immediately after a restart).
            trends = self._build_trends(use_restart_fallback=True)

            # Per-symbol HTF biases
            htf_biases = {sym: dict(b) for sym, b in self._htf_cache.items()}
            
            # Per-symbol OB state (for watchlist colored star markers).
            # Only populated when the OB Filter is enabled — otherwise
            # the UI shouldn't show stars (would be visual noise when
            # filter isn't gating anything). Single batch DB query for
            # all watchlist symbols at the configured filter TF.
            ob_filter_enabled = self._settings.get('ob_filter_enabled', False)
            ob_filter_tf = self._settings.get('ob_filter_timeframe', '1h')
            ob_states = {}
            if ob_filter_enabled and self._watchlist:
                try:
                    from storage.db_operations import get_db
                    rows = get_db().get_smc_ob_states_bulk(
                        list(self._watchlist), ob_filter_tf)
                    # Flatten to compact dict — UI only needs bias and tag.
                    # Sending the full row would inflate /api/smc/state
                    # payload by ~400 bytes per symbol on every 10s poll.
                    for sym, row in rows.items():
                        ob_states[sym] = {
                            'bias': row.get('bias'),
                            'created_by_tag': row.get('created_by_tag'),
                        }
                except Exception as e:
                    # Don't fail the whole state call on a DB hiccup —
                    # UI just won't show stars this poll, refreshes next time
                    print(f"[SMC] ob_states fetch error: {e}")
            
            return {
                'running': self._running,
                'enabled': self._settings.get('enabled', True),
                'watchlist': list(self._watchlist),
                'watchlist_sources': {s: self._get_sources().get(s, 'manual')
                                      for s in self._watchlist},
                'tradeable': list(self._tradeable),
                'settings': dict(self._settings),
                'scan_count': self._scan_count,
                'errors': self._errors,
                'cached_symbols': list(self._cache.keys()),
                'trends': trends,
                'htf_filter_active': htf_active,
                'htf_biases': htf_biases,
                'ob_filter_enabled': ob_filter_enabled,
                'ob_filter_timeframe': ob_filter_tf,
                'ob_states': ob_states,
                # Volumized OB trend per symbol — flat map keyed by
                # symbol, value 'LONG'/'SHORT'/None. Watchlist table
                # renders a column from this. Small payload — single
                # token per symbol — so no need to gate by toggle.
                'volumized_trends': {
                    sym: cache.get('trend')
                    for sym, cache in self._volumized_trend_cache.items()
                },
                'volumized_enabled': bool(self._settings.get('use_volumized_ob', True)),
                'volumized_timeframe': self._settings.get('volumized_timeframe', '1h'),
                'pending_choch': {k: {'dir': v['dir'], 'level': v['level']}
                                   for k, v in self._pending_choch.items()},
                
                # === Dedup state (Pine "1 per trend") ===
                # Per-symbol record of the last-alerted direction. Three
                # possible states the UI surfaces as a pill in the watchlist:
                #
                #   not in dict      → "🔓 OPEN" (no signal fired yet — first
                #                       signal in either direction will alert)
                #   "LONG"           → "🟢 ↑ LONG fired" (next alert must be
                #                       SHORT to pass the dedup gate)
                #   "SHORT"          → "🔴 ↓ SHORT fired" (next must be LONG)
                #
                # The pill is rendered only when `deduplicate_signals: true`
                # in settings — otherwise every signal is alerted regardless
                # of prior state, so the pill carries no useful information.
                'dedup_state': dict(self._last_signal_dir),
                'dedup_enabled': bool(self._settings.get('deduplicate_signals', True)),
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
