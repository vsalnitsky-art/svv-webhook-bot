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
            
            # === STABLE matching grid (bugfix 2026-06-10) ===
            # Previously grid = mid × MATCH_GRID_PCT recomputed every
            # update — mid drifts every tick, so int(price/grid) gave
            # DIFFERENT keys for the same wall, mass-classifying stable
            # walls as vanished+new ("phantom spoofs", inflated pct).
            # Now the grid is anchored per symbol at first sight and only
            # re-anchored when mid drifts >10% (tracked state reset then,
            # since old keys are meaningless after re-anchor).
            grid_w = st.get('grid_w')
            anchor = st.get('grid_anchor_mid')
            if grid_w is None or anchor is None or abs(mid - anchor) / anchor > 0.10:
                st['grid_w'] = grid_w = mid * MATCH_GRID_PCT
                st['grid_anchor_mid'] = mid
                st['tracked'] = {}
            
            all_walls = ((walls.get('bid_walls') or [])
                         + (walls.get('ask_walls') or []))
            top_usd = max((w.get('usd', 0) for w in all_walls), default=0)
            min_track = max(MIN_TRACK_USD, top_usd * MIN_TRACK_FRAC)
            
            current = {}
            for w in all_walls:
                usd = w.get('usd', 0)
                if usd < min_track:
                    continue
                key = (w.get('side'), int(w['price'] / grid_w))
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
            
            # Promote long-lived walls to "persistent" (once each).
            # BUGFIX 2026-06-10: previously the check was
            # (now - first_ts) >= PERSIST_MIN_LIFE with no liveness
            # requirement, so under sparse sampling (signal scanner: one
            # update per ~120s) a wall seen ONCE got promoted at the next
            # update purely because wall-clock time had passed — and its
            # vanish was then excluded from spoof counting. Net effect:
            # spoofs were structurally impossible (always 0%).
            # Correct rule: persistence = OBSERVED lifetime (last - first),
            # and the wall must still be alive (seen recently).
            for key, t in tracked.items():
                alive = key in current
                observed_life = t['last_ts'] - t['first_ts']
                if (not t['counted_persist'] and alive
                        and observed_life >= PERSIST_MIN_LIFE):
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
                # Sparse-sampling guard: if the gap since we last saw the
                # wall exceeds SPOOF_MAX_LIFE, we genuinely don't know how
                # long it lived after our last observation — classifying
                # it as a spoof would be a guess. Skip (neither bucket).
                gap = now - t['last_ts']
                if (lifetime < SPOOF_MAX_LIFE
                        and gap <= SPOOF_MAX_LIFE
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
            # === Side breakdown (2026-06-11) ===
            # Spoof tuple: (ts, usd, lifetime, price, side). Where the fake
            # liquidity is painted carries direction:
            #   spoofed ASKS = fake resistance (suppress the look of price
            #                  while accumulating)        → LONG-leaning
            #   spoofed BIDS = fake support (lure longs /
            #                  prop price before pulling) → SHORT-leaning
            # dir_score in [-1, +1]: +1 = all spoofed USD on asks (LONG),
            # -1 = all on bids (SHORT). USD-weighted, not count-weighted —
            # one $50M fake wall says more than five $100K ones.
            spoof_bid_usd = sum(s[1] for s in spoofs if s[4] == 'bid')
            spoof_ask_usd = sum(s[1] for s in spoofs if s[4] == 'ask')
            spoof_bid_n = sum(1 for s in spoofs if s[4] == 'bid')
            spoof_ask_n = sum(1 for s in spoofs if s[4] == 'ask')
            side_den = spoof_bid_usd + spoof_ask_usd
            dir_score = ((spoof_ask_usd - spoof_bid_usd) / side_den
                         if side_den > 0 else 0.0)
            recent = [{'price': p, 'usd': round(u, 0),
                       'lifetime_s': round(lt, 1), 'side': sd}
                      for (_, u, lt, p, sd) in spoofs[-5:]]
            return {
                'ok': True,
                'pct': round(pct, 1),
                'spoof_count': n_s,
                'persistent_count': n_p,
                'spoofed_usd': round(sum(s[1] for s in spoofs), 0),
                'spoof_bid_usd': round(spoof_bid_usd, 0),
                'spoof_ask_usd': round(spoof_ask_usd, 0),
                'spoof_bid_count': spoof_bid_n,
                'spoof_ask_count': spoof_ask_n,
                'dir_score': round(dir_score, 3),
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
