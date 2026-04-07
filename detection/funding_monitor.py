"""
Funding Rate Monitor v1.0 — Extreme Funding Rate Scanner

Scans all Bybit perpetual contracts every 5 minutes.
Detects coins with extreme funding rates (≤ -2% or ≥ +2%).

Extreme negative funding = shorts overpay → potential short squeeze → LONG candidate
Extreme positive funding = longs overpay → potential long squeeze → SHORT candidate

Uses existing Bybit connector (1 API call per scan for all 650+ coins).
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# Config
SCAN_INTERVAL = 300         # 5 minutes
THRESHOLD_NEGATIVE = -0.02  # -2% (Bybit returns decimal: -0.02 = -2%)
THRESHOLD_POSITIVE = 0.02   # +2%
HISTORY_DAYS = 3            # Rolling window
MAX_HISTORY_PER_DAY = 288   # 24h × 12 scans/h


class FundingMonitor:
    """Monitors Bybit funding rates for extreme values."""
    
    def __init__(self, bybit_connector=None, db=None, scan_interval: int = SCAN_INTERVAL):
        self.bybit = bybit_connector
        self.db = db
        self.scan_interval = scan_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Current state
        self._extreme_coins: List[Dict] = []
        self._all_rates: Dict[str, float] = {}  # symbol → rate
        self._last_scan: str = ''
        self._scan_count: int = 0
        self._errors: int = 0
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FundingMonitor")
        self._thread.start()
        print(f"[FUNDING] ✅ Started: every {self.scan_interval}s, "
              f"threshold: ≤{THRESHOLD_NEGATIVE*100:.1f}% / ≥+{THRESHOLD_POSITIVE*100:.1f}%")
    
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
    
    def _scan(self):
        """Fetch all tickers and filter extreme funding rates."""
        if not self.bybit:
            return
        
        try:
            tickers = self.bybit.get_tickers(category='linear')
            if not tickers:
                return
            
            now = datetime.now(timezone.utc)
            now_str = now.strftime('%Y-%m-%d %H:%M')
            
            extreme = []
            all_rates = {}
            
            for t in tickers:
                symbol = t.get('symbol', '')
                rate_str = t.get('fundingRate', '0')
                price_str = t.get('lastPrice', '0')
                volume_str = t.get('volume24h', '0')
                
                try:
                    rate = float(rate_str)
                    price = float(price_str)
                    volume = float(volume_str)
                except (ValueError, TypeError):
                    continue
                
                if not symbol or not symbol.endswith('USDT'):
                    continue
                
                all_rates[symbol] = rate
                
                # Filter extreme rates
                if rate <= THRESHOLD_NEGATIVE or rate >= THRESHOLD_POSITIVE:
                    extreme.append({
                        'symbol': symbol,
                        'rate': round(rate * 100, 4),  # Convert to percentage
                        'rate_raw': rate,
                        'price': price,
                        'volume_24h': round(volume, 0),
                        'direction': 'LONG' if rate <= THRESHOLD_NEGATIVE else 'SHORT',
                        'timestamp': now_str,
                    })
            
            # Sort by absolute rate (most extreme first)
            extreme.sort(key=lambda x: abs(x['rate']), reverse=True)
            
            with self._lock:
                self._extreme_coins = extreme
                self._all_rates = all_rates
                self._last_scan = now_str
                self._scan_count += 1
            
            # Store history
            self._store_history(extreme, now)
            
            # Cleanup old history
            if self._scan_count % 50 == 0:
                self._cleanup()
            
            # Log
            if self._scan_count <= 2 or self._scan_count % 12 == 0 or len(extreme) > 0:
                neg = len([e for e in extreme if e['rate'] < 0])
                pos = len([e for e in extreme if e['rate'] > 0])
                total = len(all_rates)
                top = extreme[0] if extreme else None
                top_str = f" | Top: {top['symbol']} {top['rate']:+.2f}%" if top else ""
                print(f"[FUNDING] #{self._scan_count}: {total} coins scanned, "
                      f"{len(extreme)} extreme ({neg} neg, {pos} pos){top_str}")
        
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[FUNDING] ⚠️ Scan error #{self._errors}: {e}")
    
    def _store_history(self, extreme: List[Dict], now: datetime):
        """Store extreme coins snapshot for daily tracking."""
        if not self.db or not extreme:
            return
        try:
            day = now.strftime('%Y-%m-%d')
            db_key = f'funding_history_{day}'
            
            history = self.db.get_setting(db_key, [])
            if not isinstance(history, list):
                history = []
            
            # Compact snapshot
            snap = {
                't': now.strftime('%H:%M'),
                'coins': [{
                    's': e['symbol'].replace('USDT', ''),
                    'r': e['rate'],
                    'd': e['direction'][0],  # 'L' or 'S'
                } for e in extreme[:20]],  # Max 20 per snapshot
            }
            history.append(snap)
            
            if len(history) > MAX_HISTORY_PER_DAY:
                history = history[-MAX_HISTORY_PER_DAY:]
            
            self.db.set_setting(db_key, history)
        except Exception as e:
            if self._scan_count <= 3:
                print(f"[FUNDING] ⚠️ Store error: {e}")
    
    def _cleanup(self):
        """Remove old history."""
        if not self.db:
            return
        try:
            for i in range(HISTORY_DAYS + 2, HISTORY_DAYS + 5):
                old_date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime('%Y-%m-%d')
                old_key = f'funding_history_{old_date}'
                try:
                    if self.db.get_setting(old_key):
                        self.db.set_setting(old_key, None)
                except:
                    pass
        except Exception as e:
            print(f"[FUNDING] ⚠️ Cleanup error: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_extreme(self) -> Dict:
        """Current extreme funding rate coins."""
        with self._lock:
            return {
                'coins': self._extreme_coins,
                'scan_count': self._scan_count,
                'last_scan': self._last_scan,
                'total_coins': len(self._all_rates),
                'errors': self._errors,
                'running': self._running,
                'threshold': {
                    'negative': THRESHOLD_NEGATIVE * 100,
                    'positive': THRESHOLD_POSITIVE * 100,
                },
            }
    
    def get_top_rates(self, limit: int = 30) -> Dict:
        """Top most extreme funding rates (both directions)."""
        with self._lock:
            if not self._all_rates:
                return {'coins': [], 'timestamp': self._last_scan}
            
            sorted_rates = sorted(self._all_rates.items(), key=lambda x: x[1])
            
            # Most negative (LONG candidates)
            most_negative = [
                {'symbol': s, 'rate': round(r * 100, 4), 'direction': 'LONG'}
                for s, r in sorted_rates[:limit]
                if r < 0
            ]
            
            # Most positive (SHORT candidates)
            most_positive = [
                {'symbol': s, 'rate': round(r * 100, 4), 'direction': 'SHORT'}
                for s, r in reversed(sorted_rates[-limit:])
                if r > 0
            ]
            
            return {
                'most_negative': most_negative[:limit],
                'most_positive': most_positive[:limit],
                'timestamp': self._last_scan,
                'total': len(self._all_rates),
            }
    
    def get_history(self, date: str = '') -> Dict:
        """History of extreme coins for a specific day."""
        if not self.db:
            return {'data': [], 'available_days': []}
        
        # Available days
        available = []
        for i in range(-1, HISTORY_DAYS + 1):
            d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime('%Y-%m-%d')
            key = f'funding_history_{d}'
            data = self.db.get_setting(key, [])
            if isinstance(data, list) and len(data) > 0:
                available.append({'date': d, 'snapshots': len(data)})
        available.sort(key=lambda x: x['date'], reverse=True)
        
        if not date and available:
            date = available[0]['date']
        
        data = self.db.get_setting(f'funding_history_{date}', [])
        if not isinstance(data, list):
            data = []
        
        return {
            'date': date,
            'data': data,
            'available_days': available,
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
