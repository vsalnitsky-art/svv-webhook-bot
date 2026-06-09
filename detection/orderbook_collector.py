"""
OrderBook Collector — WebSocket-based per-symbol depth20 collector.

Why WebSocket instead of REST:
  Binance Futures REST endpoint /fapi/v1/depth is rate-limited and the
  IP block of Render.com datacenters is currently 418-blocked (see
  HEATMAP] HTTP 418 spam in logs). WebSocket streams bypass this block
  because Binance distinguishes WS traffic from REST traffic and
  generally treats WS as cheap (no weight cost, no per-minute limit).

What we collect:
  For each requested symbol, we subscribe to
    wss://fstream.binance.com/ws/<symbol>@depth20@500ms
  which pushes the top-20 bid + top-20 ask levels every 500ms. This
  covers approximately ±0.5–2% around mid-price for major coins — the
  range where actionable order book walls live. We don't try to keep
  full-depth state (would require @depth diff streams with thousands of
  events/sec per symbol).

LRU cap + idle cleanup:
  We cap active subscriptions at MAX_ACTIVE_SYMBOLS (default 20). When
  the cap is hit, the least-recently-requested symbol is disconnected.
  We also drop any symbol that hasn't been requested in IDLE_TIMEOUT_SECS
  (default 5 min) so background WS connections don't accumulate forever.
  The frontend pulls /api/orderbook/walls every few seconds while the
  user views the page; when they navigate away the polling stops, the
  symbol goes idle, and we clean up.

Cluster algorithm:
  Adjacent depth levels (within CLUSTER_PCT of each other) get merged
  into a single "wall" so users see consolidated zones rather than 20
  individual lines. USD value = sum of (price * qty) within the cluster.
  Top-N clusters by USD on each side are returned to the UI.

Imbalance + bonus metrics:
  Same snapshot is enough to compute total bid/ask USD imbalance — a
  cheap directional pressure indicator. Returned in same payload so the
  UI can show it without a second API call.

Thread safety:
  All state mutations go through self._lock. Each symbol's WS runs in
  its own daemon thread and only writes to its own entry; the lock just
  guards the symbol dict structure itself.
"""

import json
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional

try:
    import websocket  # websocket-client>=1.6.0
except ImportError:
    websocket = None


# Hard caps and timing knobs — tuned for Render free tier where memory
# and threads are precious. Order book snapshots are ~2-4KB each.
MAX_ACTIVE_SYMBOLS = 20
IDLE_TIMEOUT_SECS = 300  # drop WS if no API request for this long
CLUSTER_PCT = 0.0005  # 0.05% — merge levels within this distance
TOP_N_WALLS = 8  # how many strongest walls per side to expose
RECONNECT_DELAY_SECS = 3
CLEANUP_INTERVAL_SECS = 30


class OrderBookCollector:
    """Singleton WS collector. Use get_orderbook_collector() to access."""
    
    def __init__(self):
        self._lock = threading.RLock()
        # OrderedDict gives us LRU semantics: move_to_end on access,
        # popitem(last=False) on eviction. Value shape:
        #   { 'symbol', 'ws', 'thread', 'snapshot', 'last_seen',
        #     'last_request', 'connect_attempts' }
        self._symbols: OrderedDict = OrderedDict()
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    def start(self):
        """Kick off the idle-cleanup background thread. Safe to call
        multiple times — second call is a no-op."""
        if self._running:
            return
        if websocket is None:
            print("[OBC] ⚠️ websocket-client not installed — collector disabled")
            return
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name='OBC-cleanup')
        self._cleanup_thread.start()
        print("[OBC] ✅ Started (WS-based, max_active=%d)" % MAX_ACTIVE_SYMBOLS)
    
    def stop(self):
        """Disconnect every active stream. Used on shutdown / tests."""
        self._running = False
        with self._lock:
            syms = list(self._symbols.keys())
        for sym in syms:
            self._disconnect(sym)
    
    # ============================================================
    # Public API — request walls for a symbol
    # ============================================================
    
    def request(self, symbol: str) -> Optional[Dict]:
        """Return latest snapshot for `symbol`. Subscribes lazily on first
        request. Returns None if no snapshot received yet (first ~500ms
        after subscription); the frontend just retries on its polling
        loop and the next call usually has data.
        """
        if not symbol or websocket is None:
            return None
        sym = symbol.upper()
        with self._lock:
            entry = self._symbols.get(sym)
            if entry is None:
                # Capacity check: if at limit, evict LRU first
                if len(self._symbols) >= MAX_ACTIVE_SYMBOLS:
                    oldest_sym = next(iter(self._symbols))
                    self._disconnect_locked(oldest_sym)
                entry = self._subscribe_locked(sym)
            entry['last_request'] = time.time()
            self._symbols.move_to_end(sym)  # mark recently used
            return entry.get('snapshot')
    
    def active_symbols(self) -> List[Dict]:
        """List currently-streaming symbols (for debug UI / status page).
        Returns lightweight summaries without the full snapshots."""
        out = []
        now = time.time()
        with self._lock:
            for sym, e in self._symbols.items():
                out.append({
                    'symbol': sym,
                    'has_snapshot': e.get('snapshot') is not None,
                    'last_seen_secs_ago': (now - e['last_seen']) if e['last_seen'] else None,
                    'last_request_secs_ago': now - e['last_request'],
                    'connect_attempts': e.get('connect_attempts', 0),
                })
        return out
    
    # ============================================================
    # Compute walls from snapshot
    # ============================================================
    
    def compute_walls(self, snapshot: Dict,
                       top_n: int = TOP_N_WALLS,
                       cluster_pct: float = CLUSTER_PCT) -> Optional[Dict]:
        """Cluster snapshot's bid+ask levels into walls and return the
        top-N by USD on each side. Also computes bid/ask imbalance which
        is a useful natural directional indicator we get "for free" from
        the same snapshot.
        
        Returns a dict ready for jsonification — no raw floats outside
        json-safe ranges, no nan/inf, no numpy types.
        """
        if not snapshot:
            return None
        bids = snapshot.get('bids') or []
        asks = snapshot.get('asks') or []
        if not bids or not asks:
            return None
        
        def cluster_levels(levels, side):
            """Merge price levels within cluster_pct into walls.
            Iterates ascending by price to keep cluster logic simple —
            we tag side at the end."""
            if not levels:
                return []
            srt = sorted(levels, key=lambda x: x[0])
            walls = []
            cur_lo = srt[0][0]
            cur_hi = srt[0][0]
            cur_qty = srt[0][1]
            for p, q in srt[1:]:
                # Distance check uses cluster's HIGH price as anchor so
                # we don't over-cluster across a widening range.
                if cur_hi > 0 and (p - cur_hi) / cur_hi <= cluster_pct:
                    cur_hi = p
                    cur_qty += q
                else:
                    walls.append((cur_lo, cur_hi, cur_qty))
                    cur_lo = p
                    cur_hi = p
                    cur_qty = q
            walls.append((cur_lo, cur_hi, cur_qty))
            # Format + compute USD per cluster
            results = []
            for lo, hi, qty in walls:
                mid = (lo + hi) / 2.0
                usd = mid * qty
                results.append({
                    'side': side,
                    'price': mid,
                    'price_lo': lo,
                    'price_hi': hi,
                    'qty': qty,
                    'usd': usd,
                })
            # Strongest first
            results.sort(key=lambda r: r['usd'], reverse=True)
            return results[:top_n]
        
        bid_walls = cluster_levels(bids, 'bid')
        ask_walls = cluster_levels(asks, 'ask')
        
        # Mid price = average of best bid + best ask (top-of-book)
        try:
            best_bid = max(b[0] for b in bids)
            best_ask = min(a[0] for a in asks)
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            spread_pct = spread / mid * 100 if mid > 0 else 0
        except Exception:
            mid = None
            best_bid = None
            best_ask = None
            spread = None
            spread_pct = None
        
        # Imbalance: how much more USD is sitting on bid vs ask
        total_bid_usd = sum(b[0] * b[1] for b in bids)
        total_ask_usd = sum(a[0] * a[1] for a in asks)
        total_usd = total_bid_usd + total_ask_usd
        imbalance_pct = ((total_bid_usd - total_ask_usd) / total_usd * 100
                         if total_usd > 0 else 0)
        
        return {
            'mid_price': mid,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'total_bid_usd': total_bid_usd,
            'total_ask_usd': total_ask_usd,
            'imbalance_pct': imbalance_pct,
            'snapshot_ts': snapshot.get('ts'),
            'snapshot_age_secs': (time.time() - snapshot['ts']) if snapshot.get('ts') else None,
        }
    
    # ============================================================
    # Subscription internals (must hold _lock)
    # ============================================================
    
    def _subscribe_locked(self, symbol: str) -> Dict:
        """Open a new WS stream for `symbol` and register entry. Must
        be called under self._lock so that the entry is published before
        the WS thread starts writing to it.
        """
        ws_url = f"wss://fstream.binance.com/ws/{symbol.lower()}@depth20@500ms"
        entry: Dict = {
            'symbol': symbol,
            'ws_url': ws_url,
            'ws': None,
            'thread': None,
            'snapshot': None,
            'last_seen': 0,
            'last_request': time.time(),
            'connect_attempts': 0,
            'should_run': True,
        }
        self._symbols[symbol] = entry
        
        # Run WS in own thread — websocket-client's run_forever is
        # blocking and not asyncio-friendly. Daemon=True so the process
        # can exit cleanly without joining.
        t = threading.Thread(
            target=self._ws_loop, args=(symbol,), daemon=True,
            name=f'OBC-{symbol}')
        entry['thread'] = t
        t.start()
        print(f"[OBC] {symbol} subscribed")
        return entry
    
    def _ws_loop(self, symbol: str):
        """Per-symbol WS runner with reconnect-on-disconnect logic.
        Exits cleanly when the entry's should_run flag is cleared, or
        when the symbol is no longer in self._symbols (evicted).
        """
        while True:
            # Check if we should still be running
            with self._lock:
                entry = self._symbols.get(symbol)
                if entry is None or not entry.get('should_run'):
                    return
                entry['connect_attempts'] = entry.get('connect_attempts', 0) + 1
                ws_url = entry['ws_url']
            
            # Define callbacks fresh on each reconnect so they close over
            # this iteration's `entry` reference — though since we look up
            # via self._symbols on every event, it doesn't really matter.
            def on_message(ws, msg):
                try:
                    data = json.loads(msg)
                    # depth20 format: { "lastUpdateId", "E", "T",
                    #                   "bids": [[p,q],...], "asks": [[p,q],...] }
                    bids_raw = data.get('bids') or data.get('b') or []
                    asks_raw = data.get('asks') or data.get('a') or []
                    bids = [(float(p), float(q)) for p, q in bids_raw if float(q) > 0]
                    asks = [(float(p), float(q)) for p, q in asks_raw if float(q) > 0]
                    if not bids or not asks:
                        return
                    snap = {
                        'bids': bids,
                        'asks': asks,
                        'ts': time.time(),
                    }
                    with self._lock:
                        e = self._symbols.get(symbol)
                        if e is not None:
                            e['snapshot'] = snap
                            e['last_seen'] = snap['ts']
                except Exception as ex:
                    # Don't spam logs on parse errors — print sparsely
                    # via attempt counter so a malformed stream doesn't
                    # flood the log.
                    pass
            
            def on_error(ws, err):
                print(f"[OBC] {symbol} WS error: {err}")
            
            def on_close(ws, code, reason):
                # Quiet by default — reconnect is automatic. Print only
                # on first close per symbol to avoid log spam.
                pass
            
            def on_open(ws):
                print(f"[OBC] {symbol} WS connected")
            
            try:
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open,
                )
                with self._lock:
                    e = self._symbols.get(symbol)
                    if e is None:
                        return
                    e['ws'] = ws
                # ping_interval keeps the conn alive through any
                # intermediate proxies / load balancers
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[OBC] {symbol} run_forever fatal: {e}")
            
            # Connection ended — check if we should reconnect
            with self._lock:
                entry = self._symbols.get(symbol)
                if entry is None or not entry.get('should_run'):
                    return
            time.sleep(RECONNECT_DELAY_SECS)
    
    def _disconnect_locked(self, symbol: str):
        """Tear down a symbol's WS. Caller must hold self._lock."""
        entry = self._symbols.pop(symbol, None)
        if not entry:
            return
        entry['should_run'] = False
        ws = entry.get('ws')
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        print(f"[OBC] {symbol} disconnected")
    
    def _disconnect(self, symbol: str):
        """Lock-acquiring wrapper around _disconnect_locked."""
        with self._lock:
            self._disconnect_locked(symbol)
    
    # ============================================================
    # Idle cleanup
    # ============================================================
    
    def _cleanup_loop(self):
        """Periodically drop symbols not requested recently. Runs as
        a daemon thread for the process lifetime.
        """
        while self._running:
            try:
                now = time.time()
                to_drop = []
                with self._lock:
                    for sym, e in self._symbols.items():
                        if now - e['last_request'] > IDLE_TIMEOUT_SECS:
                            to_drop.append(sym)
                for sym in to_drop:
                    self._disconnect(sym)
            except Exception as e:
                print(f"[OBC] cleanup err: {e}")
            time.sleep(CLEANUP_INTERVAL_SECS)


# ============================================================
# Module-level singleton
# ============================================================

_collector_instance: Optional[OrderBookCollector] = None


def get_orderbook_collector() -> OrderBookCollector:
    """Get the singleton collector. Created lazily on first call but
    NOT started — caller must call .start() once at app init."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = OrderBookCollector()
    return _collector_instance


# ============================================================
# Bybit REST deep order book (PRIMARY source since 2026-06-09)
# ============================================================
# Why this exists: Binance WS depth20 only gives the top-20 levels —
# for BTC that's ±$5 around mid (0.008%), so every wall collapses
# onto the mid-price line and the chart shows one stripe. The
# reference UX (walls at -1%, -5%, -11% from price) requires DEEP
# book data. Bybit REST /v5/market/orderbook serves up to 500
# levels per side, unauthenticated, and Bybit REST is NOT blocked
# from Render (the bot trades through it constantly).
#
# 500 levels coverage varies by symbol tick density:
#   - BTCUSDT (dense): ±0.1–0.5% — still narrow but 25x wider than depth20
#   - Altcoins (ADA, XRP, XLM...): ±2–15% — matches the reference look
#
# Caching: module-level per-symbol cache with TTL. The dashboard polls
# every 2s; TTL=2.0s means at most one Bybit call per poll cycle per
# symbol. Bybit public rate limit for this endpoint is 50 req/s — we're
# orders of magnitude below it.

import requests as _requests

_BYBIT_OB_URL = 'https://api.bybit.com/v5/market/orderbook'
_BYBIT_OB_LIMIT = 500          # max for linear category
_BYBIT_CACHE_TTL = 2.0         # seconds
_bybit_ob_cache: Dict[str, Dict] = {}   # symbol -> {'ts': float, 'snapshot': dict}
_bybit_ob_lock = threading.Lock()


def fetch_bybit_orderbook(symbol: str) -> Optional[Dict]:
    """Fetch deep order book (500 levels/side) from Bybit REST.
    
    Returns snapshot in the SAME shape the WS collector produces:
      {'bids': [(price, qty), ...], 'asks': [(price, qty), ...], 'ts': float}
    so compute_walls() can consume it unchanged. None on any failure
    (caller should fall back to the WS collector).
    """
    sym = (symbol or '').upper().strip()
    if sym.endswith('.P'):
        sym = sym[:-2]
    if not sym:
        return None
    
    now = time.time()
    with _bybit_ob_lock:
        cached = _bybit_ob_cache.get(sym)
        if cached and (now - cached['ts']) < _BYBIT_CACHE_TTL:
            return cached['snapshot']
    
    try:
        r = _requests.get(_BYBIT_OB_URL, params={
            'category': 'linear',
            'symbol': sym,
            'limit': _BYBIT_OB_LIMIT,
        }, timeout=6)
        if r.status_code != 200:
            print(f"[OBC] Bybit orderbook {sym}: HTTP {r.status_code}")
            return None
        d = r.json()
        if d.get('retCode') != 0:
            print(f"[OBC] Bybit orderbook {sym}: retCode={d.get('retCode')} "
                  f"{d.get('retMsg')}")
            return None
        res = d.get('result') or {}
        # Bybit shape: result.b = [["price","qty"],...] bids,
        #              result.a = [["price","qty"],...] asks
        bids = []
        for row in (res.get('b') or []):
            try:
                p = float(row[0]); q = float(row[1])
                if q > 0:
                    bids.append((p, q))
            except (TypeError, ValueError, IndexError):
                continue
        asks = []
        for row in (res.get('a') or []):
            try:
                p = float(row[0]); q = float(row[1])
                if q > 0:
                    asks.append((p, q))
            except (TypeError, ValueError, IndexError):
                continue
        if not bids or not asks:
            return None
        snapshot = {'bids': bids, 'asks': asks, 'ts': now}
        with _bybit_ob_lock:
            _bybit_ob_cache[sym] = {'ts': now, 'snapshot': snapshot}
            # Bound the cache so symbol-hopping doesn't grow it forever
            if len(_bybit_ob_cache) > 50:
                oldest = min(_bybit_ob_cache, key=lambda k: _bybit_ob_cache[k]['ts'])
                _bybit_ob_cache.pop(oldest, None)
        return snapshot
    except Exception as e:
        print(f"[OBC] Bybit orderbook {sym} fetch error: {e}")
        return None


# ============================================================
# Bucket-based wall detection v2 (2026-06-09)
# ============================================================
# The chain-clustering in compute_walls() degenerates on dense books:
# in BTC's book every adjacent level is closer than cluster_pct, so the
# chain never breaks and THE ENTIRE SIDE merges into one giant "wall"
# (observed live: single $18.4M bid wall = whole bid side).
#
# v2 approach, matching the reference UX:
#   1. FIXED price buckets (0.1% of mid wide) — no chaining, can't degenerate
#   2. EXCLUDE the ±0.1% zone around mid — that's routine market-maker
#      depth, not "walls"; the reference's nearest wall sits at +1.3%
#   3. SPATIAL DIVERSITY — greedy top-N selection with a minimum 0.2%
#      separation between displayed walls, so the output is a handful of
#      distinct levels spread across the book, not 8 labels on one line

def compute_walls_buckets(snapshot: Dict,
                           top_n: int = 8,
                           bucket_pct: float = 0.001,
                           exclude_near_pct: float = 0.001,
                           min_sep_pct: float = 0.002) -> Optional[Dict]:
    """Bucket-aggregate the book and return spatially-diverse top walls.
    
    Returns the same response shape as OrderBookCollector.compute_walls()
    so the API endpoint and frontend need no changes.
    """
    if not snapshot:
        return None
    bids = snapshot.get('bids') or []
    asks = snapshot.get('asks') or []
    if not bids or not asks:
        return None
    
    best_bid = max(b[0] for b in bids)
    best_ask = min(a[0] for a in asks)
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return None
    spread = best_ask - best_bid
    spread_pct = spread / mid * 100
    
    bucket_w = mid * bucket_pct
    
    def bucketize(levels, side):
        buckets: Dict[int, list] = {}
        for p, q in levels:
            # Skip the market-maker zone right at mid — it's always the
            # biggest USD concentration and it's NOT a wall, it's just
            # how order books look. Excluding it lets genuine outlier
            # clusters farther out win the top-N.
            if abs(p - mid) / mid < exclude_near_pct:
                continue
            idx = int(p // bucket_w)
            b = buckets.setdefault(idx, [0.0, 0.0])
            b[0] += q
            b[1] += p * q
        out = []
        for idx, (qty, usd) in buckets.items():
            center = (idx + 0.5) * bucket_w
            out.append({
                'side': side,
                'price': center,
                'price_lo': idx * bucket_w,
                'price_hi': (idx + 1) * bucket_w,
                'qty': qty,
                'usd': usd,
            })
        out.sort(key=lambda w: w['usd'], reverse=True)
        return out
    
    def select_diverse(cands):
        """Greedy: walk candidates from biggest USD down, accept each
        one that's at least min_sep_pct away from everything already
        accepted. Output: few distinct levels instead of a clump."""
        chosen = []
        for c in cands:
            ok = True
            for x in chosen:
                if abs(c['price'] - x['price']) / mid < min_sep_pct:
                    ok = False
                    break
            if ok:
                chosen.append(c)
                if len(chosen) >= top_n:
                    break
        return chosen
    
    bid_walls = select_diverse(bucketize(bids, 'bid'))
    ask_walls = select_diverse(bucketize(asks, 'ask'))
    
    # Totals/imbalance computed over the FULL book (including near-mid
    # zone) — the directional pressure metric should reflect everything.
    total_bid_usd = sum(p * q for p, q in bids)
    total_ask_usd = sum(p * q for p, q in asks)
    total_usd = total_bid_usd + total_ask_usd
    imbalance_pct = ((total_bid_usd - total_ask_usd) / total_usd * 100
                     if total_usd > 0 else 0)
    
    return {
        'mid_price': mid,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'spread_pct': spread_pct,
        'bid_walls': bid_walls,
        'ask_walls': ask_walls,
        'total_bid_usd': total_bid_usd,
        'total_ask_usd': total_ask_usd,
        'imbalance_pct': imbalance_pct,
        'snapshot_ts': snapshot.get('ts'),
        'snapshot_age_secs': (time.time() - snapshot['ts']) if snapshot.get('ts') else None,
    }
