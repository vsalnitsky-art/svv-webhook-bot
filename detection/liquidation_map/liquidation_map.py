"""
LiquidationMapDaemon — background scanner that polls providers, aggregates
OI deltas into buckets, marks mitigated levels, and exposes state to the
Flask API layer.

Per-tick flow (every 60s):
  1. Resolve active symbol list: BACKGROUND_SYMBOLS + on-demand (TTL'd)
  2. For each symbol, in parallel across providers:
       a. Fetch OISnapshot
       b. Pull previous snapshot from DB (we cache one tick back per provider)
       c. Compute delta → emit bucket contributions
       d. Persist new snapshot, upsert buckets
  3. For each active symbol: fetch latest klines, mitigate buckets that
     the latest candle wicked through
  4. Daily: prune buckets older than retention_days (30d default)

On-demand symbol lifecycle:
  - UI POSTs to /api/liquidation-map/request-symbol → daemon adds to set
  - First scan tick after request starts accumulating buckets
  - If symbol not re-requested for ON_DEMAND_TTL seconds (30min) it drops
    off the active set (saves API quota; reactivates on next request)
  - BACKGROUND_SYMBOLS (BTC, ETH) never drop — they're always tracked
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Set
from datetime import datetime, timezone

from detection.liquidation_map.providers import (
    OIEstimationProvider, HyperliquidProvider, OISnapshot,
)
from detection.liquidation_map.bucket_aggregator import (
    attribute_oi_delta, find_clusters, default_bucket_size,
)


BACKGROUND_SYMBOLS = ['BTCUSDT', 'ETHUSDT']

# Decay profiles for ladder levels: name → (half-life hours, window hours).
# 'tori' is the default — calibrated against the reference map the user
# trades with (most aggressive suppression of stale OI builds).
DECAY_PROFILES = {
    'full':    (8.0, 24.0),
    'fresh4h': (4.0, 24.0),
    'win12h':  (8.0, 12.0),
    'tori':    (4.0, 12.0),
}
SCAN_INTERVAL_SEC = 60
# On-demand TTL — скільки символ лишається у скані після ОСТАННЬОГО запиту.
# FF переофіксовує робочі монети (позиції/черга/funding/BTC) щотіка (~30с), тож
# короткий TTL тримає у скані ЛИШЕ реально робочі монети, а переглянуті в UI
# (get_panel) швидко випадають. Було 1800 (30 хв) — роздувало скан до десятків
# зайвих символів і памʼять. Тепер 300с (5 хв), налаштовно env
# LIQMAP_ONDEMAND_TTL_SEC. Мінімум 60с, щоб фонове сканування встигало.
try:
    ON_DEMAND_TTL_SEC = max(60, int(os.getenv('LIQMAP_ONDEMAND_TTL_SEC', '300')))
except (TypeError, ValueError):
    ON_DEMAND_TTL_SEC = 300
PARALLEL_PROVIDERS = 4              # workers in fetch pool
KLINES_LOOKBACK_BARS = 3            # bars to check for mitigation each tick
PRICE_WINDOW_PCT = 0.10             # ±10% from mark for UI bucket query
RETENTION_DAYS = 30
PRUNE_INTERVAL_SEC = 3600           # check for pruning once an hour


class LiquidationMapDaemon:
    
    def __init__(self, db=None, market_data=None):
        """market_data: optional reference to detection.market_data for klines
        fetches used by the mitigation pass. If None, mitigation is skipped
        (daemon still runs and accumulates buckets — they just never get
        marked as mitigated). Allows daemon to start before market_data is
        wired up."""
        self.db = db
        self.market_data = market_data
        
        self.providers = [
            OIEstimationProvider(),
            HyperliquidProvider(),
        ]
        
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # symbol → last_request_ts (epoch). BACKGROUND_SYMBOLS are implicitly
        # always-active, not in this dict.
        self._on_demand_ttl: Dict[str, float] = {}
        
        # Stats for the /state endpoint's data_quality block
        self._scan_count = 0
        self._errors = 0
        self._last_tick_at: Optional[float] = None
        self._symbol_first_seen: Dict[str, float] = {}
        self._last_prune_at: float = 0.0
    
    # ------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------
    
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._scan_loop, name='LiquidationMap', daemon=True)
            self._thread.start()
            print('[LIQMAP] Daemon started')
    
    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop_event.set()
    
    def is_running(self) -> bool:
        return self._running
    
    # ------------------------------------------------------------------------
    # External API — UI calls these
    # ------------------------------------------------------------------------
    
    def request_symbol(self, symbol: str) -> Dict:
        """Add (or refresh) a symbol to the active scan list. Returns info
        about whether history is being built or already available."""
        s = symbol.upper().strip()
        if not s.endswith('USDT'):
            return {'ok': False, 'reason': 'Only USDT-margined perps supported'}
        with self._lock:
            new = s not in self._on_demand_ttl and s not in BACKGROUND_SYMBOLS
            self._on_demand_ttl[s] = time.time()
            first_seen = self._symbol_first_seen.get(s)
        return {
            'ok': True,
            'symbol': s,
            'is_new': new,
            'first_seen_ts': first_seen,
            'history_hours': (time.time() - first_seen) / 3600 if first_seen else 0,
        }
    
    def get_active_symbols(self) -> List[str]:
        with self._lock:
            now = time.time()
            on_demand = [s for s, ts in self._on_demand_ttl.items()
                          if now - ts < ON_DEMAND_TTL_SEC]
            # Preserve order: background first, then on-demand
            return BACKGROUND_SYMBOLS + [s for s in on_demand
                                          if s not in BACKGROUND_SYMBOLS]
    
    def get_state(self, symbol: str, lookback_hours: int = 24,
                    include_mitigated: bool = False,
                    profile: str = 'tori') -> Dict:
        """Build the response for /api/liquidation-map/state.
        
        Returns:
          - mark_price + ts
          - events: list of individual liquidation events in the lookback
                    window (each rendered as one short dash on the chart)
          - cluster_zones: aggregated magnet zones, computed on-read by
                           summing events per (price, side) and clustering
          - data_quality: provenance + history info
          - window: the price band we filtered to
        """
        symbol = symbol.upper().strip()
        
        # Pull the latest known mark price from the last OI snapshot of any
        # provider — daemon writes one every tick.
        latest_oi = None
        for p in self.providers:
            row = self.db.liqmap_get_latest_oi(symbol, p.name)
            if row and (latest_oi is None or row['ts'] > latest_oi['ts']):
                latest_oi = row
        if not latest_oi:
            return {
                'symbol': symbol,
                'mark_price': None,
                'events': [],
                'cluster_zones': [],
                'data_quality': self._data_quality(symbol),
                'last_update': None,
            }
        
        mark_price = latest_oi['mark_price']
        # Filter to ±PRICE_WINDOW_PCT around mark price — UI doesn't need
        # events 50% away, just the active band
        min_price = mark_price * (1 - PRICE_WINDOW_PCT)
        max_price = mark_price * (1 + PRICE_WINDOW_PCT)
        
        events = self.db.liqmap_get_events(
            symbol,
            lookback_seconds=lookback_hours * 3600,
            min_price=min_price, max_price=max_price,
            include_mitigated=include_mitigated,
            limit=3000,
        )
        
        # Aggregate events into pseudo-buckets per (price, side, leverage)
        # so cluster detection (which expects that shape) can run.
        agg: Dict[tuple, Dict] = {}
        for e in events:
            if e['mitigated_at_ts']:
                continue
            key = (e['bucket_price'], e['side'], e['leverage'])
            if key not in agg:
                agg[key] = {
                    'bucket_price': e['bucket_price'],
                    'side': e['side'],
                    'leverage': e['leverage'],
                    'cumulative_usd': 0.0,
                    'first_seen_ts': e['ts'],
                    'mitigated_at_ts': None,
                }
            agg[key]['cumulative_usd'] += e['usd_added']
            if e['ts'] < agg[key]['first_seen_ts']:
                agg[key]['first_seen_ts'] = e['ts']
        agg_list = sorted(agg.values(), key=lambda b: b['bucket_price'])
        clusters = find_clusters(agg_list, mark_price)
        
        # === Ladder levels (2026-06-12, tori-style calibration) ===
        # Per-LEVEL aggregation with TIME DECAY — the visual unit of the
        # new chart. Two deliberate differences from cluster_zones:
        #   1) fine price buckets (LEVEL_GRID_PCT of mark) instead of wide
        #      multi-percent zones — a zone summing a whole band into one
        #      $480M number inflated the scale ~100× vs per-level reality;
        #   2) exponential decay (profile half-life): an OI build from 20
        #      hours ago is mostly stale — positions close, stops move.
        # Mitigated events are already excluded above.
        #
        # Decay PROFILES (2026-06-12 v2, user-selectable; quantitative
        # comparison on a uniform 24h event stream — retained share of
        # raw USD mass:  full 42% · fresh4h 24% · win12h 31% · tori 17%):
        #   full    8h half-life / 24h window — the complete map
        #   fresh4h 4h / 24h — old builds fade fast, window stays wide
        #   win12h  8h / 12h — hard cutoff, gentle fading inside
        #   tori    4h / 12h — both knobs; closest to the reference map
        prof = DECAY_PROFILES.get(profile or 'tori', DECAY_PROFILES['tori'])
        LEVEL_GRID_PCT = 0.0025      # 0.25% buckets
        LEVEL_HALFLIFE_H = prof[0]
        LEVEL_WINDOW_H = prof[1]
        LEVEL_TOP_N_PER_SIDE = 9
        now_ts = latest_oi['ts']
        grid = max(mark_price * LEVEL_GRID_PCT, 1e-12)
        window_cut = now_ts - LEVEL_WINDOW_H * 3600.0
        lvl: Dict[tuple, Dict] = {}
        for e in events:
            if e['mitigated_at_ts']:
                continue
            if e['ts'] < window_cut:
                continue
            age_h = max(0.0, (now_ts - e['ts']) / 3600.0)
            w = 0.5 ** (age_h / LEVEL_HALFLIFE_H)
            lp = (int(e['bucket_price'] / grid) + 0.5) * grid
            key = (round(lp, 12), e['side'])
            rec = lvl.get(key)
            if rec is None:
                rec = {'price': lp, 'side': e['side'],
                       'usd': 0.0, 'raw_usd': 0.0, 'newest_ts': e['ts']}
                lvl[key] = rec
            rec['usd'] += e['usd_added'] * w
            rec['raw_usd'] += e['usd_added']
            if e['ts'] > rec['newest_ts']:
                rec['newest_ts'] = e['ts']
        # Top-N per side by decayed USD; tiny residue levels are noise
        longs = sorted((r for r in lvl.values() if r['side'] == 'long'),
                       key=lambda r: -r['usd'])[:LEVEL_TOP_N_PER_SIDE]
        shorts = sorted((r for r in lvl.values() if r['side'] == 'short'),
                        key=lambda r: -r['usd'])[:LEVEL_TOP_N_PER_SIDE]
        levels = sorted(longs + shorts, key=lambda r: r['price'])
        levels = [{'price': round(r['price'], 10),
                   'usd': round(r['usd'], 0),
                   'raw_usd': round(r['raw_usd'], 0),
                   'side': r['side'],
                   'age_min': round((now_ts - r['newest_ts']) / 60.0, 0)}
                  for r in levels if r['usd'] >= 1000]
        
        return {
            'symbol': symbol,
            'mark_price': mark_price,
            'mark_price_ts': latest_oi['ts'],
            'events': events,
            'cluster_zones': clusters,
            'levels': levels,
            'decay_profile': profile if profile in DECAY_PROFILES else 'tori',
            'data_quality': self._data_quality(symbol),
            'last_update': datetime.fromtimestamp(
                latest_oi['ts'], tz=timezone.utc).isoformat(),
            'window': {
                'min_price': min_price,
                'max_price': max_price,
                'lookback_hours': lookback_hours,
            },
        }
    
    def _data_quality(self, symbol: str) -> Dict:
        """Provenance info shown to user — which providers worked recently,
        how much history exists."""
        first_seen = self._symbol_first_seen.get(symbol)
        history_hours = (time.time() - first_seen) / 3600 if first_seen else 0
        return {
            'estimation_healthy': self.providers[0].is_healthy(),
            'hyperliquid_healthy': self.providers[1].is_healthy(),
            'history_hours': round(history_hours, 1),
            # Map history to confidence label
            'confidence': ('high' if history_hours >= 4
                           else 'medium' if history_hours >= 1
                           else 'low'),
        }
    
    # ------------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------------
    
    def _scan_loop(self) -> None:
        # Stagger startup so we don't hammer APIs immediately on boot
        if self._stop_event.wait(timeout=10):
            return
        
        while self._running and not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                self._errors += 1
                if self._errors <= 10:
                    print(f'[LIQMAP] Tick error: {e}')
            
            # Periodic pruning (separate from tick error budget)
            if time.time() - self._last_prune_at > PRUNE_INTERVAL_SEC:
                try:
                    n = self.db.liqmap_prune_events(RETENTION_DAYS)
                    if n:
                        print(f'[LIQMAP] Pruned {n} stale events')
                except Exception as e:
                    print(f'[LIQMAP] Prune error: {e}')
                self._last_prune_at = time.time()
            
            if self._stop_event.wait(timeout=SCAN_INTERVAL_SEC):
                break
    
    def _tick(self) -> None:
        symbols = self.get_active_symbols()
        if not symbols:
            return
        
        now = time.time()
        with self._lock:
            for s in symbols:
                self._symbol_first_seen.setdefault(s, now)
        
        scan_start = time.time()
        
        # For each (symbol, provider) pair, fetch+process in parallel.
        with ThreadPoolExecutor(max_workers=PARALLEL_PROVIDERS) as pool:
            futures = []
            for symbol in symbols:
                for provider in self.providers:
                    futures.append(pool.submit(
                        self._process_symbol_provider, symbol, provider))
            for f in futures:
                try:
                    f.result(timeout=30)
                except Exception as e:
                    if self._errors <= 10:
                        print(f'[LIQMAP] Provider task error: {e}')
                    self._errors += 1
        
        # Mitigation pass — separate per-symbol (not per-provider) since
        # klines fetch is provider-agnostic
        if self.market_data is not None:
            for symbol in symbols:
                try:
                    self._mitigate_symbol(symbol)
                except Exception as e:
                    if self._errors <= 10:
                        print(f'[LIQMAP] Mitigation error {symbol}: {e}')
                    self._errors += 1
        
        self._scan_count += 1
        self._last_tick_at = time.time()
        dur = self._last_tick_at - scan_start
        if self._scan_count % 5 == 1:  # log every 5 ticks to avoid spam
            print(f'[LIQMAP] Tick #{self._scan_count} done in {dur:.1f}s '
                  f'({len(symbols)} symbols × {len(self.providers)} providers)')
    
    def _process_symbol_provider(self, symbol: str, provider) -> None:
        """Fetch one snapshot, compute delta vs DB prev, persist, emit
        bucket contributions. Catches its own exceptions so executor
        timeouts don't propagate."""
        try:
            snap = provider.fetch_oi_snapshot(symbol)
        except Exception as e:
            print(f'[LIQMAP] {provider.name} {symbol} fetch error: {e}')
            return
        if not snap:
            return
        
        # Pull previous snapshot from DB
        prev_row = self.db.liqmap_get_latest_oi(symbol, provider.name)
        
        # Persist current first — even if delta computation fails we have
        # a checkpoint
        self.db.liqmap_save_oi_snapshot(
            symbol=symbol, exchange=provider.name, ts=snap.ts,
            oi_usd=snap.open_interest_usd, mark_price=snap.mark_price,
            long_ratio=snap.long_ratio,
        )
        
        if not prev_row:
            # First snapshot for this symbol+provider — nothing to compare
            return
        
        # Build a synthetic OISnapshot for the prev row (aggregator interface)
        prev_snap = OISnapshot(
            symbol=symbol, exchange=provider.name,
            ts=prev_row['ts'],
            mark_price=prev_row['mark_price'],
            open_interest_usd=prev_row['oi_usd'],
            long_ratio=prev_row['long_ratio'],
        )
        
        # Time-gap sanity — if previous snapshot is >10 minutes old, skip
        # delta computation. The OI may have changed dramatically and we'd
        # attribute the whole change to "this tick" which is wrong.
        if snap.ts - prev_snap.ts > 600:
            return
        
        contributions = list(attribute_oi_delta(
            prev_snap, snap, symbol=symbol, source=provider.name,
        ))
        
        # Event-based write: each contribution is a new immutable event row.
        # Batched insert to keep per-tick DB load minimal.
        events_batch = [{
            'symbol': symbol,
            'ts': snap.ts,
            'bucket_price': c.bucket_price,
            'side': c.side,
            'leverage': c.leverage,
            'usd_added': c.usd_added,
            'source': c.source,
        } for c in contributions if c.usd_added > 0]
        if events_batch:
            self.db.liqmap_save_events_batch(events_batch)
    
    def _mitigate_symbol(self, symbol: str) -> None:
        """Check the latest klines for the symbol; mark any active bucket
        whose price was wicked through as mitigated."""
        if not self.market_data:
            return
        try:
            klines = self.market_data.fetch_klines(
                symbol, interval='1m', limit=KLINES_LOOKBACK_BARS)
        except Exception:
            return
        if not klines or len(klines) < 1:
            return
        
        # Determine the price range covered by recent candles (wick-based)
        lows = [float(k.get('l', k.get('low', 0))) for k in klines]
        highs = [float(k.get('h', k.get('high', 0))) for k in klines]
        lows = [v for v in lows if v > 0]
        highs = [v for v in highs if v > 0]
        if not lows or not highs:
            return
        
        # Mitigation band: extend wick low/high so buckets at the exact edge
        # also count as hit. Buckets entirely within the wick range get
        # marked.
        low = min(lows)
        high = max(highs)
        
        now_ts = int(time.time())
        n = self.db.liqmap_mark_events_mitigated(symbol, low, high, now_ts)
        if n:
            print(f'[LIQMAP] {symbol}: mitigated {n} events in '
                  f'[{low:.2f}, {high:.2f}]')


# ============================================================================
# Singleton
# ============================================================================

_instance: Optional[LiquidationMapDaemon] = None


def init_liquidation_map(db=None, market_data=None) -> LiquidationMapDaemon:
    """Idempotent — returns existing instance if already created."""
    global _instance
    if _instance is None:
        _instance = LiquidationMapDaemon(db=db, market_data=market_data)
    return _instance


def get_liquidation_map() -> Optional[LiquidationMapDaemon]:
    return _instance
