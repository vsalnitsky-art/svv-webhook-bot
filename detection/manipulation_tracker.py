"""
manipulation_tracker.py — spoof detection via order-book wall lifecycle.

Idea: a genuine wall is liquidity someone intends to fill — it persists.
A spoof is a large order parked to paint a fake picture (scare/lure) and
pulled before price can reach it. We observe walls on every dashboard
poll (~2s cadence) and classify their disappearance:

    SPOOF      lifetime < SPOOF_MAX_LIFE  AND  price never came close
               (wall was pulled, not consumed)
    PERSISTENT alive ≥ PERSIST_MIN_LIFE   (counted once per wall)
    CONSUMED   vanished while price was at/через the level — that's a
               legitimate fill (or mitigation), NOT manipulation

Manipulation % over a rolling 10-minute window:
    pct = spoofs / (spoofs + persistents) × 100

Honest limitations (documented, not hidden):
  - We only see walls while the dashboard polls (collector is poll-driven).
    The gauge needs ~2 minutes of warmup after the page opens / symbol
    switches, and pauses when nobody is watching.
  - "Price reached the wall" is approximated by comparing mid to the wall
    price AT VANISH TIME (±0.1%). A wick that touched the level and
    snapped back between two polls can misclassify a consumed wall as a
    spoof. At 2s polling this is rare but possible on fast markets.
  - Aggregated 3-exchange walls mean a "pull" on one venue while another
    holds shows as a size drop, not a vanish — partially masking
    single-venue spoofing. Trade-off accepted for the aggregate view.

Thread-safe singleton; update() is called from the /api/orderbook/walls
endpoint after wall computation, get_state() from the same response.
"""

import time
import threading
from collections import deque
from typing import Dict, Optional

WINDOW_SEC = 600          # rolling stats window (10 min)
SPOOF_MAX_LIFE = 60       # wall pulled faster than this → spoof candidate
PERSIST_MIN_LIFE = 90     # wall alive this long → counted as genuine
VANISH_GRACE_SEC = 6      # must be absent ~3 polls before declaring gone
PRICE_REACH_TOL = 0.001   # mid within 0.1% of wall = price reached it
MATCH_GRID_PCT = 0.002    # walls matched on a 0.2% price grid across polls
MIN_TRACK_USD = 20_000    # ignore dust below this
MIN_TRACK_FRAC = 0.10     # ...and below 10% of the current top wall


class ManipulationTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._sym: Dict[str, Dict] = {}
    
    def _state_for(self, symbol: str) -> Dict:
        st = self._sym.get(symbol)
        if st is None:
            st = {
                'tracked': {},          # grid_key -> wall lifecycle dict
                'spoofs': deque(),      # (ts, max_usd, lifetime, price, side)
                'persists': deque(),    # (ts, max_usd)
                'first_update_ts': time.time(),
                'last_update_ts': 0.0,
            }
            self._sym[symbol] = st
            # Bound symbol map (symbol-hopping)
            if len(self._sym) > 50:
                oldest = min(self._sym,
                             key=lambda k: self._sym[k]['last_update_ts'])
                if oldest != symbol:
                    self._sym.pop(oldest, None)
        return st
    
    def update(self, symbol: str, walls: Dict):
        """Feed one walls snapshot (output of compute_walls_buckets_v3)."""
        if not walls or not walls.get('mid_price'):
            return
        mid = walls['mid_price']
        now = time.time()
        
        with self._lock:
            st = self._state_for(symbol)
            st['last_update_ts'] = now
            
            all_walls = ((walls.get('bid_walls') or [])
                         + (walls.get('ask_walls') or []))
            top_usd = max((w.get('usd', 0) for w in all_walls), default=0)
            min_track = max(MIN_TRACK_USD, top_usd * MIN_TRACK_FRAC)
            
            grid = mid * MATCH_GRID_PCT
            current = {}
            for w in all_walls:
                usd = w.get('usd', 0)
                if usd < min_track:
                    continue
                key = (w.get('side'), int(w['price'] / grid))
                # Keep the bigger if two walls land on one grid cell
                if key not in current or usd > current[key]['usd']:
                    current[key] = w
            
            tracked = st['tracked']
            # Update / insert live walls
            for key, w in current.items():
                t = tracked.get(key)
                if t:
                    t['last_ts'] = now
                    t['price'] = w['price']
                    if w['usd'] > t['max_usd']:
                        t['max_usd'] = w['usd']
                else:
                    tracked[key] = {
                        'first_ts': now, 'last_ts': now,
                        'price': w['price'], 'side': w.get('side'),
                        'max_usd': w['usd'], 'counted_persist': False,
                    }
            
            # Promote long-lived walls to "persistent" (once each)
            for t in tracked.values():
                if (not t['counted_persist']
                        and (now - t['first_ts']) >= PERSIST_MIN_LIFE):
                    t['counted_persist'] = True
                    st['persists'].append((now, t['max_usd']))
            
            # Detect vanished walls (absent for > grace period)
            gone_keys = [k for k, t in tracked.items()
                         if k not in current
                         and (now - t['last_ts']) > VANISH_GRACE_SEC]
            for k in gone_keys:
                t = tracked.pop(k)
                lifetime = t['last_ts'] - t['first_ts']
                price_reached = (abs(mid - t['price']) / mid) < PRICE_REACH_TOL
                if (lifetime < SPOOF_MAX_LIFE
                        and not price_reached
                        and not t['counted_persist']):
                    st['spoofs'].append((now, t['max_usd'], lifetime,
                                          t['price'], t['side']))
            
            # Prune windows
            cutoff = now - WINDOW_SEC
            while st['spoofs'] and st['spoofs'][0][0] < cutoff:
                st['spoofs'].popleft()
            while st['persists'] and st['persists'][0][0] < cutoff:
                st['persists'].popleft()
    
    def get_state(self, symbol: str) -> Dict:
        """Rolling manipulation stats for the gauge."""
        now = time.time()
        with self._lock:
            st = self._sym.get(symbol)
            if st is None:
                return {'ok': True, 'pct': 0, 'spoof_count': 0,
                        'persistent_count': 0, 'spoofed_usd': 0,
                        'window_min': WINDOW_SEC // 60,
                        'warming_up': True, 'observed_sec': 0,
                        'recent_spoofs': []}
            
            observed = now - st['first_update_ts']
            spoofs = list(st['spoofs'])
            persists = list(st['persists'])
            n_s = len(spoofs)
            n_p = len(persists)
            denom = n_s + n_p
            pct = (n_s / denom * 100) if denom > 0 else 0
            recent = [{'price': p, 'usd': round(u, 0),
                       'lifetime_s': round(lt, 1), 'side': sd}
                      for (_, u, lt, p, sd) in spoofs[-5:]]
            return {
                'ok': True,
                'pct': round(pct, 1),
                'spoof_count': n_s,
                'persistent_count': n_p,
                'spoofed_usd': round(sum(s[1] for s in spoofs), 0),
                'window_min': WINDOW_SEC // 60,
                'warming_up': observed < 120,
                'observed_sec': round(observed, 0),
                'recent_spoofs': recent,
            }


_instance: Optional[ManipulationTracker] = None
_instance_lock = threading.Lock()


def get_manipulation_tracker() -> ManipulationTracker:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ManipulationTracker()
    return _instance
