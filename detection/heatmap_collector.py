"""
Multi-Symbol Liquidity Heatmap Collector v1.0

Fetches Binance Futures depth for a configurable list of symbols every 60s.
For each symbol, stores a compact depth profile (bid+ask clusters within ±3%
of mid-price) to the `sob_liq_heatmap_profiles` table.

Each profile is one row per (symbol, ts). The frontend Heatmap tab aggregates
these into a time × price grid for CoinGlass-style visualization.

This collector is SEPARATE from `detection/liquidity_map.py` which handles
BTC-only wall detection used by trading logic. Both can run side-by-side —
they hit the same public Binance endpoint but otherwise share nothing.

Symbol list is configurable via DB setting `heatmap_symbols`
(comma-separated, default "BTCUSDT,ETHUSDT,SOLUSDT").
Sample budget: 3 symbols × 1 req/min = 3 req/min, each weight 20 → 60/min
against the 2400/min Binance Futures limit. Plenty of headroom for 5-10 symbols.
"""

import json
import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ========================================
# CONFIG
# ========================================

BINANCE_DEPTH_URL = 'https://fapi.binance.com/fapi/v1/depth'
DEPTH_LIMIT = 1000               # max levels per side
HEATMAP_CLUSTER_SIZE = 5         # $5 buckets — matches old liquidity_map convention
MAX_DISTANCE_PCT = 3.0           # ±3% of mid-price only — keeps row size compact
MIN_CLUSTER_USD = 50_000         # drop very thin clusters to reduce noise

SCAN_INTERVAL = 60               # seconds per round-robin cycle
RETENTION_DAYS = 7               # cleanup target
CLEANUP_EVERY = 60               # run cleanup every N scan cycles (≈ once an hour)

DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

# Symbol-specific cluster size override. For low-priced coins $5 is too wide;
# scale down so the grid still has resolution. Falls through to a 0.05%
# heuristic for symbols not in this map.
SYMBOL_CLUSTER_OVERRIDES = {
    'BTCUSDT': 5,
    'ETHUSDT': 1,
    'SOLUSDT': 0.1,
    'BNBUSDT': 0.5,
    'XRPUSDT': 0.001,
    'DOGEUSDT': 0.0005,
}


def _cluster_size_for(symbol: str, mid_price: float) -> float:
    """Return a sensible bucket size for a symbol given current price.
    Pick the override if present, else 0.05% of mid_price rounded to
    a clean step. Floor at $0.0001 to avoid zero division on tiny prices.
    """
    if symbol in SYMBOL_CLUSTER_OVERRIDES:
        return SYMBOL_CLUSTER_OVERRIDES[symbol]
    step = max(mid_price * 0.0005, 0.0001)
    # Round to a "nice" step (1, 2, 5, 10, ...)
    import math
    exp = math.floor(math.log10(step))
    mantissa = step / (10 ** exp)
    if mantissa < 1.5:
        nice = 1
    elif mantissa < 3.5:
        nice = 2
    elif mantissa < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


class HeatmapCollector:
    """Multi-symbol liquidity heatmap snapshotter."""

    def __init__(self, db=None, scan_interval: int = SCAN_INTERVAL):
        self.db = db
        self.scan_interval = scan_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._symbols: List[str] = list(DEFAULT_SYMBOLS)
        self._reload_symbols_from_db()

        self._scan_count: int = 0
        self._error_count: int = 0
        self._last_scan_ts: Optional[str] = None
        self._last_results: Dict[str, Dict] = {}   # symbol → {'price', 'bids_n', 'asks_n', 'ts'}

        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'VSV-Bot/1.0'})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        # Check DB toggle. Enabled by default — there's no reason to keep this
        # off once the feature exists, but we honor an explicit "0" for ops.
        if self.db and self.db.get_setting('heatmap_enabled', '1') != '1':
            print("[HEATMAP] Disabled in DB (heatmap_enabled=0), not starting")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                          name="HeatmapCollector")
        self._thread.start()
        print(f"[HEATMAP] ✅ Started — symbols={self._symbols}")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _reload_symbols_from_db(self):
        """Read `heatmap_symbols` from DB, fall back to defaults."""
        if not self.db:
            return
        try:
            raw = self.db.get_setting('heatmap_symbols', None)
            if raw:
                if isinstance(raw, list):
                    items = raw
                else:
                    items = [s.strip().upper() for s in str(raw).split(',') if s.strip()]
                items = [s if s.endswith('USDT') else f'{s}USDT' for s in items]
                # De-dup, keep order
                seen = set()
                ordered = []
                for s in items:
                    if s not in seen:
                        seen.add(s)
                        ordered.append(s)
                if ordered:
                    self._symbols = ordered
        except Exception as e:
            print(f"[HEATMAP] symbols reload error: {e}")

    def get_symbols(self) -> List[str]:
        with self._lock:
            return list(self._symbols)

    def set_symbols(self, symbols: List[str]) -> List[str]:
        """Replace the symbol list. Persists to DB. Returns the applied list."""
        cleaned = []
        for s in symbols:
            s = str(s).strip().upper()
            if not s:
                continue
            if not s.endswith('USDT'):
                s = f'{s}USDT'
            if s not in cleaned:
                cleaned.append(s)
        if not cleaned:
            cleaned = list(DEFAULT_SYMBOLS)
        with self._lock:
            self._symbols = cleaned
        if self.db:
            try:
                self.db.set_setting('heatmap_symbols', ','.join(cleaned))
            except Exception as e:
                print(f"[HEATMAP] set_symbols persist error: {e}")
        return cleaned

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------
    def _loop(self):
        # Small startup delay so we don't dogpile with other workers
        time.sleep(2)
        while self._running:
            cycle_start = time.time()
            try:
                # Reload symbols every cycle — lets the user adjust without restart
                self._reload_symbols_from_db()
                symbols = self.get_symbols()
                for sym in symbols:
                    if not self._running:
                        break
                    self._scan_one(sym)
                    # Small stagger to avoid bursting
                    time.sleep(0.5)
                self._scan_count += 1
                if self._scan_count <= 2 or self._scan_count % 10 == 0:
                    print(f"[HEATMAP] cycle #{self._scan_count} done "
                          f"({len(symbols)} symbols)")
                # Periodic cleanup
                if self._scan_count % CLEANUP_EVERY == 0:
                    self._cleanup()
            except Exception as e:
                self._error_count += 1
                if self._error_count <= 5 or self._error_count % 20 == 0:
                    print(f"[HEATMAP] ⚠️ cycle error #{self._error_count}: "
                          f"{type(e).__name__}: {e}")
            # Sleep the remainder of the cycle
            elapsed = time.time() - cycle_start
            sleep_for = max(1.0, self.scan_interval - elapsed)
            for _ in range(int(sleep_for)):
                if not self._running:
                    break
                time.sleep(1)

    def _scan_one(self, symbol: str):
        """Fetch depth for one symbol, build a compact profile, persist."""
        try:
            resp = self._session.get(
                BINANCE_DEPTH_URL,
                params={'symbol': symbol, 'limit': DEPTH_LIMIT},
                timeout=8
            )
            if resp.status_code != 200:
                if self._error_count <= 10:
                    print(f"[HEATMAP] {symbol}: HTTP {resp.status_code}")
                return
            data = resp.json()
            bids = data.get('bids') or []
            asks = data.get('asks') or []
            if not bids or not asks:
                return
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            if mid_price <= 0:
                return

            bucket_size = _cluster_size_for(symbol, mid_price)
            max_dist = mid_price * (MAX_DISTANCE_PCT / 100)

            bid_clusters: Dict[float, float] = {}
            for p_str, q_str in bids:
                p = float(p_str)
                if mid_price - p > max_dist:
                    continue
                q = float(q_str)
                vol = p * q
                bucket = round(p / bucket_size) * bucket_size
                bid_clusters[bucket] = bid_clusters.get(bucket, 0) + vol

            ask_clusters: Dict[float, float] = {}
            for p_str, q_str in asks:
                p = float(p_str)
                if p - mid_price > max_dist:
                    continue
                q = float(q_str)
                vol = p * q
                bucket = round(p / bucket_size) * bucket_size
                ask_clusters[bucket] = ask_clusters.get(bucket, 0) + vol

            # Compact format: prices as floats (kept precision), volumes in $K rounded
            # Filter out thin clusters
            bid_arr = sorted(
                [[round(p, 6), round(v / 1000)] for p, v in bid_clusters.items()
                 if v >= MIN_CLUSTER_USD],
                key=lambda x: x[0]
            )
            ask_arr = sorted(
                [[round(p, 6), round(v / 1000)] for p, v in ask_clusters.items()
                 if v >= MIN_CLUSTER_USD],
                key=lambda x: x[0]
            )

            if not bid_arr and not ask_arr:
                return

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')

            if self.db:
                self.db.add_liq_heatmap_profile(
                    symbol=symbol,
                    ts=now,
                    mid_price=mid_price,
                    bid_data=json.dumps(bid_arr, separators=(',', ':')),
                    ask_data=json.dumps(ask_arr, separators=(',', ':')),
                )

            with self._lock:
                self._last_scan_ts = now_str
                self._last_results[symbol] = {
                    'price': round(mid_price, 6),
                    'bids_n': len(bid_arr),
                    'asks_n': len(ask_arr),
                    'ts': now_str,
                    'bucket_size': bucket_size,
                }
        except requests.exceptions.RequestException as e:
            self._error_count += 1
            if self._error_count <= 5:
                print(f"[HEATMAP] {symbol} HTTP err: {type(e).__name__}: {e}")
        except Exception as e:
            self._error_count += 1
            if self._error_count <= 5:
                print(f"[HEATMAP] {symbol} scan err: {type(e).__name__}: {e}")

    def _cleanup(self):
        if not self.db:
            return
        try:
            n = self.db.cleanup_liq_heatmap_profiles(retention_days=RETENTION_DAYS)
            if n > 0:
                print(f"[HEATMAP] 🧹 cleanup: removed {n} old profiles")
        except Exception as e:
            print(f"[HEATMAP] cleanup error: {e}")

    # ------------------------------------------------------------------
    # Read API — used by the Flask endpoint
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        """Lightweight status snapshot — what symbols are configured and
        when each was last scanned. Used by the UI symbol selector.
        """
        with self._lock:
            return {
                'symbols': list(self._symbols),
                'last_scan_ts': self._last_scan_ts,
                'scan_count': self._scan_count,
                'error_count': self._error_count,
                'last_results': dict(self._last_results),
            }

    def get_heatmap_data(self, symbol: str, hours: int = 24,
                           bucket_minutes: int = 5) -> Dict:
        """Aggregate stored profiles into a time × price grid for rendering.

        Returns a dict with:
            symbol, hours, snapshots, current_price, max_volume,
            time_buckets: ['HH:MM', ...],
            rows: [{price, cells: [v1, v2, ...]}],
            price_min, price_max, price_line: [price_per_bucket, ...]
        """
        empty = self._empty_response(symbol, hours)
        if not self.db:
            return empty

        rows = self.db.get_liq_heatmap_profiles(symbol=symbol, hours=hours)
        if not rows:
            return empty

        # Bucket snapshots into N-minute groups; within a bucket, take max
        # per-price (not sum) — multiple snapshots reflect persistent liquidity,
        # not stacked liquidity.
        bucket_minutes = max(1, min(bucket_minutes, 60))
        bucket_map: Dict[str, Dict[float, float]] = {}
        price_by_bucket: Dict[str, float] = {}
        snapshots_used = 0

        for row in rows:
            ts_str = row.get('ts', '') or ''
            if not ts_str or len(ts_str) < 16:
                continue
            try:
                # 'YYYY-MM-DDTHH:MM:SS' from to_dict isoformat
                date_part = ts_str[:10]
                hh = int(ts_str[11:13])
                mm = int(ts_str[14:16])
                bucket_min = (mm // bucket_minutes) * bucket_minutes
                bucket_key = f"{date_part} {hh:02d}:{bucket_min:02d}"
            except Exception:
                continue

            try:
                bid_clusters = json.loads(row.get('bid_data') or '[]')
                ask_clusters = json.loads(row.get('ask_data') or '[]')
            except Exception:
                continue

            cell = bucket_map.setdefault(bucket_key, {})
            for p, v in bid_clusters:
                cur = cell.get(p, 0)
                if v > cur:
                    cell[p] = v
            for p, v in ask_clusters:
                cur = cell.get(p, 0)
                if v > cur:
                    cell[p] = v
            price_by_bucket[bucket_key] = float(row.get('mid_price') or 0)
            snapshots_used += 1

        if not bucket_map:
            return empty

        sorted_buckets = sorted(bucket_map.keys())
        all_prices = set()
        for cell in bucket_map.values():
            all_prices.update(cell.keys())
        if not all_prices:
            return empty
        sorted_prices = sorted(all_prices, reverse=True)  # high → low so top of chart = high

        # Build rows (one per price level) with the cell list aligned to sorted_buckets
        out_rows = []
        max_v = 0
        for p in sorted_prices:
            cells = []
            for bk in sorted_buckets:
                v = bucket_map[bk].get(p, 0)
                cells.append(v)
                if v > max_v:
                    max_v = v
            out_rows.append({'price': p, 'cells': cells})

        # Time labels — strip the date for compactness
        time_labels = [bk[11:] if len(bk) >= 16 else bk for bk in sorted_buckets]
        price_line = [price_by_bucket.get(bk, 0) for bk in sorted_buckets]

        current_price = price_line[-1] if price_line else (sorted_prices[0] if sorted_prices else 0)

        return {
            'symbol': symbol,
            'hours': hours,
            'snapshots': snapshots_used,
            'current_price': current_price,
            'price_min': min(sorted_prices) if sorted_prices else 0,
            'price_max': max(sorted_prices) if sorted_prices else 0,
            'time_buckets': time_labels,
            'rows': out_rows,
            'price_line': price_line,
            'max_volume': max_v,
        }

    def _empty_response(self, symbol: str, hours: int) -> Dict:
        return {
            'symbol': symbol,
            'hours': hours,
            'snapshots': 0,
            'current_price': 0,
            'price_min': 0,
            'price_max': 0,
            'time_buckets': [],
            'rows': [],
            'price_line': [],
            'max_volume': 0,
        }


# ============================================================
# Singleton
# ============================================================

_instance: Optional[HeatmapCollector] = None


def get_heatmap_collector() -> Optional[HeatmapCollector]:
    return _instance


def init_heatmap_collector(db=None) -> HeatmapCollector:
    global _instance
    if _instance is not None:
        return _instance
    _instance = HeatmapCollector(db=db)
    return _instance
