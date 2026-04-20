"""
Whale Tape v1.0 — Large Trades Feed

Polls Binance aggTrades for BTCUSDT every 10 seconds and filters trades
above configurable USD threshold (default $100K). Stores rolling buffer
of last N large trades.

Binance aggTrade format:
  a: aggregate trade ID
  p: price (string)
  q: quantity (string)
  T: timestamp (ms)
  m: true = buyer is maker (SELL aggressor = red)
     false = buyer is taker (BUY aggressor = green)

Stats computed:
  - Total buy vs sell volume (aggressor-based)
  - Largest single trade
  - Average trade size
  - Net delta (buy - sell)
"""

import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional


# Config
BINANCE_AGG_URL = 'https://fapi.binance.com/fapi/v1/aggTrades'
SYMBOL = 'BTCUSDT'
POLL_INTERVAL = 10                   # seconds
AGG_LIMIT = 1000                     # max per poll (Binance cap)
MIN_TRADE_USD = 100_000              # default whale threshold
MAX_BUFFER = 500                     # keep last N large trades in memory
REQUEST_TIMEOUT = 15
DB_KEY = 'whale_tape_settings'


class WhaleTape:
    """Polls Binance aggTrades and filters whale-sized trades."""
    
    def __init__(self, db=None, symbol: str = SYMBOL):
        self.db = db
        self.symbol = symbol
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
        
        # Rolling buffer of large trades
        self._trades: List[Dict] = []
        self._last_agg_id: int = 0  # dedupe
        
        # Stats
        self._scan_count = 0
        self._errors = 0
        self._total_fetched = 0
        self._total_whales = 0
        
        # Load persisted threshold
        self._min_usd = MIN_TRADE_USD
        if db:
            try:
                stored = db.get_setting(DB_KEY, {})
                if isinstance(stored, dict):
                    self._min_usd = int(stored.get('min_usd', MIN_TRADE_USD))
            except:
                pass
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="WhaleTape")
        self._thread.start()
        print(f"[WHALES] ✅ Started: {self.symbol}, every {POLL_INTERVAL}s, "
              f"threshold ${self._min_usd:,}")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _loop(self):
        print("[WHALES] 🧵 Thread started")
        time.sleep(5)  # wait for other modules to initialize
        while self._running:
            try:
                self._poll()
            except Exception as e:
                self._errors += 1
                if self._errors <= 5:
                    print(f"[WHALES] Poll error: {e}")
            
            # Sleep in 1-sec chunks for responsive shutdown
            for _ in range(POLL_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)
    
    def _poll(self):
        resp = self._session.get(
            BINANCE_AGG_URL,
            params={'symbol': self.symbol, 'limit': AGG_LIMIT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, list):
            return
        
        self._scan_count += 1
        self._total_fetched += len(data)
        
        new_whales = []
        max_agg_id = self._last_agg_id
        
        for t in data:
            try:
                agg_id = int(t.get('a', 0))
                # Skip trades we've already seen (dedupe across polls)
                if agg_id <= self._last_agg_id:
                    continue
                if agg_id > max_agg_id:
                    max_agg_id = agg_id
                
                price = float(t.get('p', 0))
                qty = float(t.get('q', 0))
                usd = price * qty
                
                if usd < self._min_usd:
                    continue
                
                is_maker_buyer = bool(t.get('m', False))
                # m=true → buyer is maker → seller was aggressor → SELL trade
                # m=false → buyer is taker → buyer was aggressor → BUY trade
                side = 'SELL' if is_maker_buyer else 'BUY'
                
                ts_ms = int(t.get('T', 0))
                
                new_whales.append({
                    'id': agg_id,
                    'price': round(price, 2) if price > 10 else price,
                    'qty': round(qty, 4),
                    'usd': round(usd),
                    'side': side,
                    't': ts_ms,
                    'time_str': datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%H:%M:%S'),
                })
            except Exception:
                continue
        
        if max_agg_id > self._last_agg_id:
            self._last_agg_id = max_agg_id
        
        if new_whales:
            self._total_whales += len(new_whales)
            with self._lock:
                # Prepend newest (they're already sorted ascending by id)
                new_whales.sort(key=lambda w: w['id'], reverse=True)
                self._trades = new_whales + self._trades
                if len(self._trades) > MAX_BUFFER:
                    self._trades = self._trades[:MAX_BUFFER]
        
        # Log occasionally
        if self._scan_count <= 2 or self._scan_count % 30 == 0:
            print(f"[WHALES] #{self._scan_count}: "
                  f"fetched={self._total_fetched}, whales={self._total_whales}, "
                  f"buffer={len(self._trades)}")
    
    # ========================================
    # Settings
    # ========================================
    
    def set_threshold(self, usd: int) -> bool:
        try:
            usd = max(10_000, min(int(usd), 10_000_000))
            with self._lock:
                self._min_usd = usd
                if self.db:
                    self.db.set_setting(DB_KEY, {'min_usd': usd})
            print(f"[WHALES] Threshold updated: ${usd:,}")
            return True
        except Exception as e:
            print(f"[WHALES] set_threshold error: {e}")
            return False
    
    def get_threshold(self) -> int:
        return self._min_usd
    
    # ========================================
    # Query
    # ========================================
    
    def get_state(self, limit: int = 100, window_minutes: int = 60) -> Dict:
        """Return recent trades + aggregated stats for given time window."""
        with self._lock:
            trades_snapshot = list(self._trades)
        
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (window_minutes * 60 * 1000)
        
        # Filter by time window for stats
        windowed = [t for t in trades_snapshot if t['t'] >= cutoff_ms]
        
        buy_vol = sum(t['usd'] for t in windowed if t['side'] == 'BUY')
        sell_vol = sum(t['usd'] for t in windowed if t['side'] == 'SELL')
        total_vol = buy_vol + sell_vol
        net_delta = buy_vol - sell_vol
        
        buy_count = sum(1 for t in windowed if t['side'] == 'BUY')
        sell_count = sum(1 for t in windowed if t['side'] == 'SELL')
        
        largest = max(windowed, key=lambda t: t['usd']) if windowed else None
        
        avg_usd = total_vol / len(windowed) if windowed else 0
        buy_pct = (buy_vol / total_vol * 100) if total_vol > 0 else 50
        
        # Return most recent N for display
        display_trades = trades_snapshot[:limit]
        
        return {
            'symbol': self.symbol,
            'threshold': self._min_usd,
            'trades': display_trades,
            'stats': {
                'window_minutes': window_minutes,
                'count': len(windowed),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'buy_volume': round(buy_vol),
                'sell_volume': round(sell_vol),
                'total_volume': round(total_vol),
                'net_delta': round(net_delta),
                'buy_pct': round(buy_pct, 1),
                'avg_usd': round(avg_usd),
                'largest': largest,
            },
            'buffer_size': len(trades_snapshot),
            'scan_count': self._scan_count,
            'errors': self._errors,
            'running': self._running,
        }


# Singleton
_instance: Optional[WhaleTape] = None


def get_whale_tape() -> Optional[WhaleTape]:
    return _instance


def init_whale_tape(db=None, symbol: str = SYMBOL) -> WhaleTape:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = WhaleTape(db=db, symbol=symbol)
    return _instance
