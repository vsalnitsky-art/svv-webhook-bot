"""
Funding Rate Monitor v2.0 — Watchlist-Based Tracker

When a coin's funding rate hits extreme threshold (≤-2% or ≥+2%):
  1. Coin added to watchlist
  2. Track its funding rate every scan for 3 days
  3. Show rate range, trend, current value
  4. After 3 days — auto-remove with all history

One API call per scan (all 650 coins). Scan every 5 minutes.
"""

import time
import math
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# Config
SCAN_INTERVAL = 300         # 5 minutes
THRESHOLD_PCT = 2.0         # ±2% triggers watchlist
WATCH_DAYS = 3              # Keep tracking for 3 days
DB_KEY = 'funding_watchlist'


class FundingMonitor:
    """Tracks coins with extreme funding rates for 3 days."""
    
    def __init__(self, bybit_connector=None, db=None, scan_interval: int = SCAN_INTERVAL):
        self.bybit = bybit_connector
        self.db = db
        self.scan_interval = scan_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Watchlist: {symbol: {first_seen, trigger_rate, direction, rates: [{t, r}...]}}
        self._watchlist: Dict[str, Dict] = {}
        self._scan_count: int = 0
        self._total_coins: int = 0
        self._errors: int = 0
        
        # Load from DB
        self._load_watchlist()
    
    def _load_watchlist(self):
        """Restore watchlist from DB."""
        if not self.db:
            return
        try:
            saved = self.db.get_setting(DB_KEY, {})
            if isinstance(saved, dict):
                self._watchlist = saved
                if self._watchlist:
                    print(f"[FUNDING] 🔄 Restored {len(self._watchlist)} tracked coins")
        except Exception as e:
            print(f"[FUNDING] ⚠️ Load error: {e}")
    
    def _save_watchlist(self):
        """Save watchlist to DB."""
        if not self.db:
            return
        try:
            self.db.set_setting(DB_KEY, self._watchlist)
        except Exception as e:
            if self._scan_count <= 3:
                print(f"[FUNDING] ⚠️ Save error: {e}")
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FundingMonitor")
        self._thread.start()
        print(f"[FUNDING] ✅ Started: every {self.scan_interval}s, "
              f"threshold: ±{THRESHOLD_PCT}%, watch: {WATCH_DAYS}d, "
              f"tracked: {len(self._watchlist)}")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[FUNDING] Stopped")
    
    def _loop(self):
        print(f"[FUNDING] 🧵 Scan thread started")
        try:
            self._scan()
            while self._running:
                time.sleep(self.scan_interval)
                if self._running:
                    self._scan()
        except Exception as e:
            print(f"[FUNDING] 💀 Thread crashed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # CORE SCAN
    # ========================================
    
    def _scan(self):
        if not self.bybit:
            return
        
        try:
            tickers = self.bybit.get_tickers(category='linear')
            if not tickers:
                return
            
            now = datetime.now(timezone.utc)
            now_str = now.strftime('%Y-%m-%d %H:%M')
            self._scan_count += 1
            
            # Parse all rates
            rates: Dict[str, Dict] = {}
            for t in tickers:
                symbol = t.get('symbol', '')
                if not symbol.endswith('USDT'):
                    continue
                try:
                    rate = float(t.get('fundingRate', '0'))
                    price = float(t.get('lastPrice', '0'))
                    vol = float(t.get('volume24h', '0'))
                except (ValueError, TypeError):
                    continue
                rates[symbol] = {'rate': rate, 'price': price, 'vol': vol}
            
            self._total_coins = len(rates)
            threshold = THRESHOLD_PCT / 100  # 0.02
            
            new_added = 0
            
            with self._lock:
                # 1. Check for NEW extreme coins → add to watchlist
                for symbol, data in rates.items():
                    rate = data['rate']
                    if symbol not in self._watchlist:
                        if rate <= -threshold or rate >= threshold:
                            direction = 'LONG' if rate < 0 else 'SHORT'
                            self._watchlist[symbol] = {
                                'first_seen': now_str,
                                'trigger_rate': round(rate * 100, 4),
                                'direction': direction,
                                'price_at_trigger': data['price'],
                                'rates': [],
                            }
                            new_added += 1
                
                # 2. Update rates for ALL tracked coins
                for symbol in list(self._watchlist.keys()):
                    coin = self._watchlist[symbol]
                    
                    if symbol in rates:
                        r = rates[symbol]
                        coin['rates'].append({
                            't': now_str[11:16],  # HH:MM
                            'r': round(r['rate'] * 100, 4),
                            'p': round(r['price'], 6),
                        })
                        
                        # Keep max ~864 points (3 days × 288 per day at 5min intervals)
                        if len(coin['rates']) > 864:
                            coin['rates'] = coin['rates'][-864:]
                
                # 3. Remove coins older than WATCH_DAYS
                cutoff = (now - timedelta(days=WATCH_DAYS)).strftime('%Y-%m-%d %H:%M')
                expired = [s for s, c in self._watchlist.items() if c.get('first_seen', '') < cutoff]
                for s in expired:
                    del self._watchlist[s]
            
            # Save to DB
            self._save_watchlist()
            
            # Log
            if self._scan_count <= 2 or self._scan_count % 12 == 0 or new_added > 0:
                print(f"[FUNDING] #{self._scan_count}: {self._total_coins} coins, "
                      f"{len(self._watchlist)} tracked"
                      f"{f', +{new_added} new' if new_added else ''}"
                      f"{f', -{len(expired)} expired' if expired else ''}")
        
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[FUNDING] ⚠️ Error #{self._errors}: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_watchlist(self) -> Dict:
        """Full watchlist with stats for each coin."""
        with self._lock:
            coins = []
            now = datetime.now(timezone.utc)
            
            for symbol, data in self._watchlist.items():
                rates = data.get('rates', [])
                
                # Calculate stats
                current_rate = rates[-1]['r'] if rates else data.get('trigger_rate', 0)
                current_price = rates[-1]['p'] if rates else data.get('price_at_trigger', 0)
                
                rate_values = [r['r'] for r in rates] if rates else [current_rate]
                min_rate = min(rate_values)
                max_rate = max(rate_values)
                
                # Time tracking
                first_seen = data.get('first_seen', '')
                try:
                    fs = datetime.strptime(first_seen, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                    hours_tracked = (now - fs).total_seconds() / 3600
                    expires_at = fs + timedelta(days=WATCH_DAYS)
                    hours_left = max(0, (expires_at - now).total_seconds() / 3600)
                except:
                    hours_tracked = 0
                    hours_left = WATCH_DAYS * 24
                
                # Price change since trigger
                trigger_price = data.get('price_at_trigger', 0)
                price_change = 0
                if trigger_price and current_price:
                    price_change = round((current_price - trigger_price) / trigger_price * 100, 2)
                
                coins.append({
                    'symbol': symbol,
                    'direction': data.get('direction', ''),
                    'trigger_rate': data.get('trigger_rate', 0),
                    'current_rate': current_rate,
                    'min_rate': min_rate,
                    'max_rate': max_rate,
                    'current_price': current_price,
                    'price_change': price_change,
                    'hours_tracked': round(hours_tracked, 1),
                    'hours_left': round(hours_left, 1),
                    'data_points': len(rates),
                    'first_seen': first_seen,
                    'rates': rates[-60:],  # Last 60 points for mini-chart
                })
            
            # Sort: newest first
            coins.sort(key=lambda x: x['first_seen'], reverse=True)
            
            return {
                'coins': coins,
                'total_tracked': len(coins),
                'scan_count': self._scan_count,
                'total_coins': self._total_coins,
                'errors': self._errors,
                'running': self._running,
                'threshold': THRESHOLD_PCT,
                'watch_days': WATCH_DAYS,
            }


# Singleton
_instance: Optional[FundingMonitor] = None

def get_funding_monitor() -> Optional[FundingMonitor]:
    return _instance

def init_funding_monitor(bybit_connector=None, db=None) -> FundingMonitor:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = FundingMonitor(bybit_connector=bybit_connector, db=db)
    return _instance
