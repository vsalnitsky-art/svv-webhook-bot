"""
BTC Volume Flow Monitor v1.0

Fetches Binance Futures 1-min klines every 60s.
Binance klines include taker_buy_volume → we get buy vs sell split.

Calculates:
  - Buy/Sell volume for 5min, 15min, 1h, 4h windows
  - CVD (Cumulative Volume Delta) = cumulative(buy - sell)
  - Volume spikes (current vs average)
  - Buy/Sell ratio trend

One request per scan: GET /fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=240
"""

import time
import threading
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

BINANCE_KLINE_URL = 'https://fapi.binance.com/fapi/v1/klines'
SYMBOL = 'BTCUSDT'
SCAN_INTERVAL = 60          # Every minute
KLINE_LIMIT = 240           # 4 hours of 1-min candles
DB_KEY_PREFIX = 'vol_flow_'
HISTORY_DAYS = 3


class VolumeFlow:
    
    def __init__(self, db=None, scan_interval: int = SCAN_INTERVAL):
        self.db = db
        self.scan_interval = scan_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
        
        # Current state
        self._candles: List[Dict] = []  # Last 240 1-min candles
        self._price: float = 0
        self._scan_count: int = 0
        self._errors: int = 0
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VolumeFlow")
        self._thread.start()
        print(f"[VOL FLOW] ✅ Started: {SYMBOL}, every {self.scan_interval}s")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _loop(self):
        print("[VOL FLOW] 🧵 Scan thread started")
        try:
            self._scan()
            while self._running:
                time.sleep(self.scan_interval)
                if self._running:
                    self._scan()
        except Exception as e:
            print(f"[VOL FLOW] 💀 Thread crashed: {e}")
            import traceback
            traceback.print_exc()
    
    def _scan(self):
        try:
            resp = self._session.get(
                BINANCE_KLINE_URL,
                params={'symbol': SYMBOL, 'interval': '1m', 'limit': KLINE_LIMIT},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
            
            if not raw or not isinstance(raw, list):
                return
            
            # Parse klines: [open_time, O, H, L, C, vol, close_time, quote_vol, trades, taker_buy_vol, taker_buy_quote_vol, ...]
            candles = []
            for k in raw:
                try:
                    total_vol = float(k[7])       # quote asset volume (USD)
                    taker_buy = float(k[10])      # taker buy quote volume (USD)
                    taker_sell = total_vol - taker_buy
                    close_price = float(k[4])
                    ts = int(k[0]) // 1000        # Unix seconds
                    
                    candles.append({
                        'ts': ts,
                        'p': close_price,
                        'v': round(total_vol),
                        'b': round(taker_buy),
                        's': round(taker_sell),
                    })
                except (ValueError, IndexError):
                    continue
            
            if not candles:
                return
            
            with self._lock:
                self._candles = candles
                self._price = candles[-1]['p']
                self._scan_count += 1
            
            # Store bias history for chart
            self._store_snapshot(candles)
            
            if self._scan_count <= 1 or self._scan_count % 30 == 0:
                s = self._calc_window(candles, 60)
                print(f"[VOL FLOW] #{self._scan_count}: {len(candles)} candles, "
                      f"1h Buy ${s['buy']/1e6:.0f}M / Sell ${s['sell']/1e6:.0f}M "
                      f"({s['buy_pct']:.0f}%/{s['sell_pct']:.0f}%)")
        
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[VOL FLOW] ⚠️ Error #{self._errors}: {e}")
    
    def _calc_window(self, candles: List[Dict], minutes: int) -> Dict:
        """Calculate buy/sell stats for last N minutes."""
        recent = candles[-minutes:] if len(candles) >= minutes else candles
        
        total_buy = sum(c['b'] for c in recent)
        total_sell = sum(c['s'] for c in recent)
        total = total_buy + total_sell
        
        buy_pct = (total_buy / total * 100) if total > 0 else 50
        sell_pct = 100 - buy_pct
        
        # CVD for this window
        cvd = total_buy - total_sell
        
        # Volume per minute average
        avg_vol = total / len(recent) if recent else 0
        
        # Current 1-min volume vs average (spike detection)
        last_vol = recent[-1]['v'] if recent else 0
        spike = (last_vol / avg_vol) if avg_vol > 0 else 1
        
        # Determine signal
        if buy_pct >= 60:
            signal = 'BUYERS'
        elif sell_pct >= 60:
            signal = 'SELLERS'
        else:
            signal = 'NEUTRAL'
        
        return {
            'buy': round(total_buy),
            'sell': round(total_sell),
            'total': round(total),
            'buy_pct': round(buy_pct, 1),
            'sell_pct': round(sell_pct, 1),
            'cvd': round(cvd),
            'signal': signal,
            'spike': round(spike, 1),
            'avg_vol_min': round(avg_vol),
        }
    
    def _store_snapshot(self, candles: List[Dict]):
        """Store 1-min snapshot for daily chart."""
        if not self.db or not candles:
            return
        try:
            last = candles[-1]
            # Calculate 5-min buy%
            w5 = self._calc_window(candles, 5)
            
            now = datetime.now(timezone.utc)
            day = now.strftime('%Y-%m-%d')
            db_key = f'{DB_KEY_PREFIX}{day}'
            
            history = self.db.get_setting(db_key, [])
            if not isinstance(history, list):
                history = []
            
            history.append({
                't': now.strftime('%H:%M'),
                'bp': round(w5['buy_pct']),  # 5-min buy %
                'p': last['p'],
                'cvd': w5['cvd'],
            })
            
            if len(history) > 1440:
                history = history[-1440:]
            
            self.db.set_setting(db_key, history)
            
            # Cleanup old days
            if self._scan_count % 60 == 0:
                for i in range(HISTORY_DAYS + 2, HISTORY_DAYS + 5):
                    old = (now - timedelta(days=i)).strftime('%Y-%m-%d')
                    try:
                        if self.db.get_setting(f'{DB_KEY_PREFIX}{old}'):
                            self.db.set_setting(f'{DB_KEY_PREFIX}{old}', None)
                    except:
                        pass
        except Exception as e:
            if self._scan_count <= 3:
                print(f"[VOL FLOW] ⚠️ Store error: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_summary(self) -> Dict:
        """Full summary for dashboard."""
        with self._lock:
            if not self._candles:
                return {'running': self._running, 'has_data': False,
                        'scan_count': self._scan_count}
            
            candles = self._candles
            
            w5 = self._calc_window(candles, 5)
            w15 = self._calc_window(candles, 15)
            w60 = self._calc_window(candles, 60)
            w240 = self._calc_window(candles, 240)
            
            # Overall signal from multiple timeframes
            signals = [w5['signal'], w15['signal'], w60['signal']]
            buyers = signals.count('BUYERS')
            sellers = signals.count('SELLERS')
            
            if buyers >= 2:
                overall = 'BUYERS DOMINANT'
            elif sellers >= 2:
                overall = 'SELLERS DOMINANT'
            else:
                overall = 'MIXED'
            
            # Volume spike (last 1-min vs 1h average)
            spike = w60['spike']
            spike_alert = spike >= 3.0
            
            return {
                'running': self._running,
                'has_data': True,
                'price': self._price,
                'scan_count': self._scan_count,
                'errors': self._errors,
                'windows': {
                    '5m': w5,
                    '15m': w15,
                    '1h': w60,
                    '4h': w240,
                },
                'overall_signal': overall,
                'spike': spike,
                'spike_alert': spike_alert,
                'cvd_1h': w60['cvd'],
            }
    
    def get_history(self, date: str = '') -> Dict:
        """Buy% history for chart."""
        if not self.db:
            return {'data': [], 'available_days': []}
        
        available = []
        for i in range(-1, HISTORY_DAYS + 1):
            d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime('%Y-%m-%d')
            data = self.db.get_setting(f'{DB_KEY_PREFIX}{d}', [])
            if isinstance(data, list) and len(data) > 0:
                available.append({'date': d, 'points': len(data)})
        available.sort(key=lambda x: x['date'], reverse=True)
        
        if not date and available:
            date = available[0]['date']
        
        data = self.db.get_setting(f'{DB_KEY_PREFIX}{date}', [])
        if not isinstance(data, list):
            data = []
        
        return {
            'date': date,
            'data': data,
            'available_days': available,
        }


# Singleton
_instance: Optional[VolumeFlow] = None

def get_volume_flow() -> Optional[VolumeFlow]:
    return _instance

def init_volume_flow(db=None) -> VolumeFlow:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = VolumeFlow(db=db)
    return _instance
