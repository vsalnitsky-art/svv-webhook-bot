"""
BTC Liquidity Map v1.0 — Order Book Wall Detector

Fetches Binance BTCUSDT depth every 60s via public REST API.
Detects significant bid/ask walls. Stores 3-day rolling history.
Self-contained: uses only requests, no binance SDK required.

Zero impact on Bybit (separate API, separate rate limits).
"""

import time
import json
import threading
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


# ========================================
# CONFIG
# ========================================

BINANCE_DEPTH_URL = 'https://fapi.binance.com/fapi/v1/depth'  # Futures, not spot
SYMBOL = 'BTCUSDT'
DEPTH_LIMIT = 1000              # Max depth levels per side
CLUSTER_SIZE = 50               # Group by $50 price buckets (tighter)
WALL_THRESHOLD = 3.0            # Wall = volume > 3× average cluster
MIN_WALL_USD = 100_000          # Minimum $100K to be a wall
MAX_DISTANCE_PCT = 3.0          # Only walls within 3% of price
MAX_WALLS_PER_SIDE = 10         # Max walls to store per snapshot
SCAN_INTERVAL = 60              # Seconds between scans
HISTORY_DAYS = 3                # Rolling window
MAX_SNAPSHOTS = 4320            # 3d × 24h × 60m


class LiquidityMap:
    """BTC Liquidity Map — detects and tracks order book walls."""
    
    def __init__(self, db=None, scan_interval: int = SCAN_INTERVAL):
        self.db = db
        self.scan_interval = scan_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Current state
        self._current: Dict = {
            'bid': [], 'ask': [], 'price': 0,
            'timestamp': '', 'scan_count': 0, 'errors': 0,
        }
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="LiquidityMap"
        )
        self._thread.start()
        print(f"[LIQ MAP] ✅ Started: {SYMBOL}, every {self.scan_interval}s, "
              f"history={HISTORY_DAYS}d, clusters=${CLUSTER_SIZE}")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[LIQ MAP] Stopped")
    
    def _loop(self):
        """Background scan loop with crash protection."""
        print(f"[LIQ MAP] 🧵 Scan thread started (tid={threading.current_thread().ident})")
        try:
            self._scan()  # Immediate first scan
            while self._running:
                time.sleep(self.scan_interval)
                if self._running:
                    self._scan()
        except Exception as e:
            print(f"[LIQ MAP] 💀 Scan thread crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[LIQ MAP] 🧵 Scan thread exited")
    
    # ========================================
    # CORE SCAN
    # ========================================
    
    def _scan(self):
        """Fetch order book → detect walls → store snapshot."""
        try:
            resp = self._session.get(
                BINANCE_DEPTH_URL,
                params={'symbol': SYMBOL, 'limit': DEPTH_LIMIT},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            if not bids or not asks:
                print(f"[LIQ MAP] ⚠️ Empty orderbook: bids={len(bids)}, asks={len(asks)}")
                return
            
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            
            bid_walls = self._detect_walls(bids, 'bid', mid_price)
            ask_walls = self._detect_walls(asks, 'ask', mid_price)
            
            now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            
            with self._lock:
                self._current['bid'] = [self._wall_dict(w) for w in bid_walls]
                self._current['ask'] = [self._wall_dict(w) for w in ask_walls]
                self._current['price'] = round(mid_price, 2)
                self._current['timestamp'] = now_str
                self._current['scan_count'] += 1
                sc = self._current['scan_count']
            
            # Debug: first 3 scans — show raw cluster stats
            if sc <= 3:
                bid_clusters = self._get_cluster_stats(bids, mid_price)
                ask_clusters = self._get_cluster_stats(asks, mid_price)
                print(f"[LIQ MAP] 🔍 DEBUG #{sc}: price=${mid_price:,.0f}, "
                      f"bids={len(bids)} levels, asks={len(asks)} levels")
                print(f"[LIQ MAP] 🔍 Bid clusters: {len(bid_clusters)}, "
                      f"max=${bid_clusters[0][1]/1e3:.0f}K @ ${bid_clusters[0][0]:,.0f}, "
                      f"avg=${sum(c[1] for c in bid_clusters)/len(bid_clusters)/1e3:.0f}K" if bid_clusters else "none")
                print(f"[LIQ MAP] 🔍 Ask clusters: {len(ask_clusters)}, "
                      f"max=${ask_clusters[0][1]/1e3:.0f}K @ ${ask_clusters[0][0]:,.0f}, "
                      f"avg=${sum(c[1] for c in ask_clusters)/len(ask_clusters)/1e3:.0f}K" if ask_clusters else "none")
                # Show top 5 bid clusters
                for i, (cp, vol) in enumerate(bid_clusters[:5]):
                    avg = sum(c[1] for c in bid_clusters) / len(bid_clusters) if bid_clusters else 1
                    print(f"[LIQ MAP] 🔍   BID #{i+1}: ${cp:,.0f} = ${vol/1e3:.0f}K ({vol/avg:.1f}× avg)")
                for i, (cp, vol) in enumerate(ask_clusters[:5]):
                    avg = sum(c[1] for c in ask_clusters) / len(ask_clusters) if ask_clusters else 1
                    print(f"[LIQ MAP] 🔍   ASK #{i+1}: ${cp:,.0f} = ${vol/1e3:.0f}K ({vol/avg:.1f}× avg)")
            
            # Store to DB
            self._store(bid_walls, ask_walls, mid_price, now_str)
            
            # Cleanup hourly
            if sc % 60 == 0:
                self._cleanup()
            
            # Log first scan and every 10th
            if sc <= 2 or sc % 10 == 0:
                tb = sum(w[1] for w in bid_walls) / 1e6
                ta = sum(w[1] for w in ask_walls) / 1e6
                print(f"[LIQ MAP] #{sc}: ${mid_price:,.0f} | "
                      f"Bid: {len(bid_walls)} walls (${tb:.1f}M) | "
                      f"Ask: {len(ask_walls)} walls (${ta:.1f}M)")
        
        except requests.exceptions.RequestException as e:
            with self._lock:
                self._current['errors'] += 1
            ec = self._current['errors']
            if ec <= 5 or ec % 10 == 0:
                print(f"[LIQ MAP] ⚠️ HTTP error #{ec}: {type(e).__name__}: {e}")
        except Exception as e:
            with self._lock:
                self._current['errors'] += 1
            print(f"[LIQ MAP] ❌ Scan error: {type(e).__name__}: {e}")
    
    # ========================================
    # WALL DETECTION
    # ========================================
    
    def _detect_walls(self, orders: List, side: str, price: float) -> List[tuple]:
        """Detect walls. Returns list of (cluster_price, volume_usd, strength)."""
        clusters: Dict[float, float] = {}
        
        for p_str, q_str in orders:
            p = float(p_str)
            q = float(q_str)
            vol = p * q
            bucket = round(p / CLUSTER_SIZE) * CLUSTER_SIZE
            clusters[bucket] = clusters.get(bucket, 0) + vol
        
        if not clusters:
            return []
        
        avg = sum(clusters.values()) / len(clusters)
        if avg <= 0:
            return []
        
        walls = []
        for cp, vol in clusters.items():
            strength = vol / avg
            dist = abs(cp - price) / price * 100
            
            if strength >= WALL_THRESHOLD and vol >= MIN_WALL_USD and dist <= MAX_DISTANCE_PCT:
                walls.append((cp, vol, strength))
        
        walls.sort(key=lambda w: w[1], reverse=True)
        return walls[:MAX_WALLS_PER_SIDE]
    
    @staticmethod
    def _wall_dict(wall: tuple) -> Dict:
        return {
            'price': wall[0],
            'volume_usd': round(wall[1], 0),
            'strength': round(wall[2], 1),
        }
    
    def _get_cluster_stats(self, orders: List, price: float) -> List[tuple]:
        """Get sorted clusters for debug logging."""
        clusters: Dict[float, float] = {}
        for p_str, q_str in orders:
            p = float(p_str)
            q = float(q_str)
            vol = p * q
            bucket = round(p / CLUSTER_SIZE) * CLUSTER_SIZE
            clusters[bucket] = clusters.get(bucket, 0) + vol
        
        # Sort by volume descending
        sorted_c = sorted(clusters.items(), key=lambda x: x[1], reverse=True)
        return sorted_c
    
    # ========================================
    # DB STORAGE
    # ========================================
    
    def _store(self, bid_walls: List[tuple], ask_walls: List[tuple],
               price: float, ts: str):
        if not self.db:
            return
        try:
            history = self.db.get_setting('liq_map_history', [])
            if not isinstance(history, list):
                history = []
            
            # Compact snapshot: price in $, volume in $K
            snap = {
                't': ts,
                'p': round(price, 0),
                'b': [[round(w[0], 0), round(w[1] / 1000, 0)] for w in bid_walls[:5]],
                'a': [[round(w[0], 0), round(w[1] / 1000, 0)] for w in ask_walls[:5]],
            }
            history.append(snap)
            
            if len(history) > MAX_SNAPSHOTS:
                history = history[-MAX_SNAPSHOTS:]
            
            self.db.set_setting('liq_map_history', history)
        except Exception as e:
            print(f"[LIQ MAP] ⚠️ Store error: {e}")
    
    def _cleanup(self):
        if not self.db:
            return
        try:
            history = self.db.get_setting('liq_map_history', [])
            if not isinstance(history, list):
                return
            cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d %H:%M')
            before = len(history)
            history = [s for s in history if s.get('t', '') >= cutoff]
            if before != len(history):
                self.db.set_setting('liq_map_history', history)
                print(f"[LIQ MAP] 🧹 Cleaned {before - len(history)} old snapshots")
        except Exception as e:
            print(f"[LIQ MAP] ⚠️ Cleanup error: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_current(self) -> Dict:
        """Current walls + metadata."""
        with self._lock:
            result = dict(self._current)
            result['thread_alive'] = self._thread.is_alive() if self._thread else False
            return result
    
    def get_persistent_walls(self) -> Dict:
        """Walls that persisted over time (institutional levels)."""
        if not self.db:
            return {'bid': [], 'ask': [], 'total_snapshots': 0, 'history_hours': 0}
        
        try:
            history = self.db.get_setting('liq_map_history', [])
            if not isinstance(history, list) or len(history) < 10:
                return {'bid': [], 'ask': [], 'total_snapshots': len(history) if isinstance(history, list) else 0, 'history_hours': 0}
            
            total = len(history)
            bid_c: Dict[float, Dict] = {}
            ask_c: Dict[float, Dict] = {}
            
            for snap in history:
                ts = snap.get('t', '')
                for p, v in snap.get('b', []):
                    p = float(p)
                    if p not in bid_c:
                        bid_c[p] = {'n': 0, 'vol': 0, 'first': ts, 'last': ts}
                    bid_c[p]['n'] += 1
                    bid_c[p]['vol'] += float(v)
                    bid_c[p]['last'] = ts
                
                for p, v in snap.get('a', []):
                    p = float(p)
                    if p not in ask_c:
                        ask_c[p] = {'n': 0, 'vol': 0, 'first': ts, 'last': ts}
                    ask_c[p]['n'] += 1
                    ask_c[p]['vol'] += float(v)
                    ask_c[p]['last'] = ts
            
            def fmt(counts, side):
                out = []
                for price, d in counts.items():
                    pct = d['n'] / total * 100
                    if pct < 5:  # Must appear in ≥5% of snapshots
                        continue
                    hours = d['n'] * self.scan_interval / 3600
                    out.append({
                        'price': price,
                        'avg_volume_k': round(d['vol'] / d['n'], 0),
                        'persistence_pct': round(pct, 1),
                        'hours_alive': round(hours, 1),
                        'appearances': d['n'],
                        'first_seen': d['first'],
                        'last_seen': d['last'],
                        'side': side,
                    })
                out.sort(key=lambda w: w['appearances'], reverse=True)
                return out[:10]
            
            return {
                'bid': fmt(bid_c, 'bid'),
                'ask': fmt(ask_c, 'ask'),
                'total_snapshots': total,
                'history_hours': round(total * self.scan_interval / 3600, 1),
            }
        except Exception as e:
            print(f"[LIQ MAP] ⚠️ Persistent error: {e}")
            return {'bid': [], 'ask': [], 'total_snapshots': 0, 'history_hours': 0}
    
    def get_summary(self) -> Dict:
        """Compact summary for dashboard widget."""
        current = self.get_current()
        persistent = self.get_persistent_walls()
        
        return {
            'price': current.get('price', 0),
            'timestamp': current.get('timestamp', ''),
            'scan_count': current.get('scan_count', 0),
            'errors': current.get('errors', 0),
            'running': self._running,
            'current_bid': current.get('bid', []),
            'current_ask': current.get('ask', []),
            'persistent_bid': persistent.get('bid', []),
            'persistent_ask': persistent.get('ask', []),
            'history_hours': persistent.get('history_hours', 0),
            'total_snapshots': persistent.get('total_snapshots', 0),
        }


# ========================================
# Singleton
# ========================================

_instance: Optional[LiquidityMap] = None

def get_liquidity_map() -> Optional[LiquidityMap]:
    return _instance

def init_liquidity_map(db=None) -> LiquidityMap:
    global _instance
    # Always create fresh instance (handles Gunicorn worker restarts)
    if _instance is not None:
        _instance.stop()
    _instance = LiquidityMap(db=db)
    return _instance
