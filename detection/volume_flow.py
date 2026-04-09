"""
BTC Volume Flow Monitor v2.0 — Trade Signal Generator

Fetches Binance Futures 1-min klines every 60s (taker buy/sell volumes).
Generates LONG/SHORT signals with confidence % based on:
  - Multi-TF buy/sell dominance (5m, 15m, 1h, 4h)
  - CVD (Cumulative Volume Delta) trend
  - Price vs CVD divergence (accumulation/distribution)
  - Volume spikes

Telegram alert on signal change (LONG→SHORT or vice versa).
"""

import time
import threading
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

BINANCE_KLINE_URL = 'https://fapi.binance.com/fapi/v1/klines'
SYMBOL = 'BTCUSDT'
SCAN_INTERVAL = 60
KLINE_LIMIT = 240
DB_KEY_PREFIX = 'vol_flow_'
HISTORY_DAYS = 3


class VolumeFlow:
    
    def __init__(self, db=None, notifier=None, scan_interval: int = SCAN_INTERVAL):
        self.db = db
        self.notifier = notifier
        self.scan_interval = scan_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
        
        self._candles: List[Dict] = []
        self._price: float = 0
        self._scan_count: int = 0
        self._errors: int = 0
        
        # Signal state (avoid spam: alert once per direction change)
        self._last_signal_dir: str = ''  # 'LONG' or 'SHORT' or ''
        self._last_signal_time: str = ''
        self._signal: Dict = {}
    
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
            
            candles = []
            for k in raw:
                try:
                    total_vol = float(k[7])
                    taker_buy = float(k[10])
                    taker_sell = total_vol - taker_buy
                    close_price = float(k[4])
                    ts = int(k[0]) // 1000
                    candles.append({
                        'ts': ts, 'p': close_price,
                        'v': round(total_vol),
                        'b': round(taker_buy), 's': round(taker_sell),
                    })
                except (ValueError, IndexError):
                    continue
            
            if not candles:
                return
            
            with self._lock:
                self._candles = candles
                self._price = candles[-1]['p']
                self._scan_count += 1
                self._signal = self._calc_signal(candles)
            
            # Check for Telegram alert
            self._check_alert()
            
            # Store history
            self._store_snapshot(candles)
            
            if self._scan_count <= 1 or self._scan_count % 30 == 0:
                s = self._calc_window(candles, 60)
                sig = self._signal
                print(f"[VOL FLOW] #{self._scan_count}: "
                      f"1h Buy {s['buy_pct']:.0f}%/Sell {s['sell_pct']:.0f}% | "
                      f"Signal: {sig.get('direction','-')} {sig.get('confidence',0)}%")
        
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[VOL FLOW] ⚠️ Error #{self._errors}: {e}")
    
    # ========================================
    # SIGNAL CALCULATION
    # ========================================
    
    def _calc_signal(self, candles: List[Dict]) -> Dict:
        """
        Calculate trade signal with confidence %.
        
        Scoring (max 100):
          +25: 5m buy/sell dominant (≥60%)
          +25: 15m buy/sell dominant (≥60%)
          +25: 1h buy/sell dominant (≥55%)
          +15: 4h confirms direction (≥55%)
          +10: Volume spike ≥2× with directional bias
          +15: CVD trend confirms (rising for LONG, falling for SHORT)
          +20: Price/CVD divergence (accumulation/distribution)
          
        Direction determined by majority of scoring components.
        """
        w5 = self._calc_window(candles, 5)
        w15 = self._calc_window(candles, 15)
        w60 = self._calc_window(candles, 60)
        w240 = self._calc_window(candles, 240)
        
        long_score = 0
        short_score = 0
        reasons = []
        
        # 1. Multi-TF buy/sell dominance
        if w5['buy_pct'] >= 60:
            long_score += 25
            reasons.append(f"5m Buyers {w5['buy_pct']:.0f}%")
        elif w5['sell_pct'] >= 60:
            short_score += 25
            reasons.append(f"5m Sellers {w5['sell_pct']:.0f}%")
        
        if w15['buy_pct'] >= 60:
            long_score += 25
            reasons.append(f"15m Buyers {w15['buy_pct']:.0f}%")
        elif w15['sell_pct'] >= 60:
            short_score += 25
            reasons.append(f"15m Sellers {w15['sell_pct']:.0f}%")
        
        if w60['buy_pct'] >= 55:
            long_score += 25
            reasons.append(f"1h Buyers {w60['buy_pct']:.0f}%")
        elif w60['sell_pct'] >= 55:
            short_score += 25
            reasons.append(f"1h Sellers {w60['sell_pct']:.0f}%")
        
        # 2. 4h confirmation
        if w240['buy_pct'] >= 55:
            long_score += 15
            reasons.append(f"4h Buyers {w240['buy_pct']:.0f}%")
        elif w240['sell_pct'] >= 55:
            short_score += 15
            reasons.append(f"4h Sellers {w240['sell_pct']:.0f}%")
        
        # 3. Volume spike with direction
        spike = w60['spike']
        if spike >= 2.0:
            last5 = candles[-5:]
            spike_buy = sum(c['b'] for c in last5)
            spike_sell = sum(c['s'] for c in last5)
            if spike_buy > spike_sell * 1.3:
                long_score += 10
                reasons.append(f"Spike {spike:.1f}× Buy")
            elif spike_sell > spike_buy * 1.3:
                short_score += 10
                reasons.append(f"Spike {spike:.1f}× Sell")
        
        # 4. CVD trend (last 15 min)
        if len(candles) >= 15:
            cvd_values = []
            running = 0
            for c in candles[-15:]:
                running += (c['b'] - c['s'])
                cvd_values.append(running)
            
            cvd_start = cvd_values[0]
            cvd_end = cvd_values[-1]
            if cvd_end > cvd_start + abs(cvd_start) * 0.1:
                long_score += 15
                reasons.append("CVD rising")
            elif cvd_end < cvd_start - abs(cvd_start) * 0.1:
                short_score += 15
                reasons.append("CVD falling")
        
        # 5. DIVERGENCE — most powerful signal
        if len(candles) >= 15:
            price_start = candles[-15]['p']
            price_end = candles[-1]['p']
            price_change = (price_end - price_start) / price_start * 100
            
            cvd_15m = sum(c['b'] - c['s'] for c in candles[-15:])
            
            # Price falling + CVD rising = accumulation → LONG
            if price_change < -0.1 and cvd_15m > 0:
                long_score += 20
                reasons.append(f"⚡ Accumulation (P {price_change:+.2f}%, CVD +)")
            # Price rising + CVD falling = distribution → SHORT
            elif price_change > 0.1 and cvd_15m < 0:
                short_score += 20
                reasons.append(f"⚡ Distribution (P {price_change:+.2f}%, CVD -)")
        
        # Determine direction and confidence
        total = long_score + short_score
        if total == 0:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': []}
        
        if long_score > short_score:
            confidence = min(95, round(long_score / 1.15))
            direction = 'LONG'
        elif short_score > long_score:
            confidence = min(95, round(short_score / 1.15))
            direction = 'SHORT'
        else:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': reasons}
        
        return {
            'direction': direction,
            'confidence': confidence,
            'long_score': long_score,
            'short_score': short_score,
            'reasons': reasons,
            'price': self._price,
        }
    
    def _check_alert(self):
        """Send Telegram on direction change with confidence ≥65%."""
        sig = self._signal
        if not sig or sig.get('confidence', 0) < 65:
            return
        
        direction = sig.get('direction', '')
        if not direction or direction == 'NEUTRAL':
            return
        
        if direction == self._last_signal_dir:
            return
        
        self._last_signal_dir = direction
        self._last_signal_time = datetime.now(timezone.utc).strftime('%H:%M')
        
        if not self.notifier:
            print(f"[VOL FLOW] 🔔 Signal: {direction} {sig['confidence']}% (no TG)")
            return
        
        try:
            icon = '🟢' if direction == 'LONG' else '🔴'
            reasons_str = '\n'.join(f"  • {r}" for r in sig.get('reasons', []))
            
            msg = (
                f"{icon} <b>BTC VOLUME SIGNAL: {direction} {sig['confidence']}%</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 BTC ${self._price:,.2f}\n"
                f"\n"
                f"📊 <b>Reasons:</b>\n"
                f"{reasons_str}\n"
                f"\n"
                f"Long score: {sig.get('long_score', 0)} | "
                f"Short score: {sig.get('short_score', 0)}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⏱ {self._last_signal_time} UTC"
            )
            self.notifier.send_message(msg)
            print(f"[VOL FLOW] 📨 TG: {direction} {sig['confidence']}%")
        except Exception as e:
            print(f"[VOL FLOW] ⚠️ Alert error: {e}")
    
    # ========================================
    # WINDOW CALCULATION
    # ========================================
    
    def _calc_window(self, candles: List[Dict], minutes: int) -> Dict:
        recent = candles[-minutes:] if len(candles) >= minutes else candles
        
        total_buy = sum(c['b'] for c in recent)
        total_sell = sum(c['s'] for c in recent)
        total = total_buy + total_sell
        
        buy_pct = (total_buy / total * 100) if total > 0 else 50
        sell_pct = 100 - buy_pct
        cvd = total_buy - total_sell
        avg_vol = total / len(recent) if recent else 0
        last_vol = recent[-1]['v'] if recent else 0
        spike = (last_vol / avg_vol) if avg_vol > 0 else 1
        
        if buy_pct >= 60:
            signal = 'BUYERS'
        elif sell_pct >= 60:
            signal = 'SELLERS'
        else:
            signal = 'NEUTRAL'
        
        return {
            'buy': round(total_buy), 'sell': round(total_sell),
            'total': round(total),
            'buy_pct': round(buy_pct, 1), 'sell_pct': round(sell_pct, 1),
            'cvd': round(cvd), 'signal': signal,
            'spike': round(spike, 1), 'avg_vol_min': round(avg_vol),
        }
    
    # ========================================
    # STORAGE
    # ========================================
    
    def _store_snapshot(self, candles: List[Dict]):
        if not self.db or not candles:
            return
        try:
            w5 = self._calc_window(candles, 5)
            now = datetime.now(timezone.utc)
            day = now.strftime('%Y-%m-%d')
            db_key = f'{DB_KEY_PREFIX}{day}'
            
            history = self.db.get_setting(db_key, [])
            if not isinstance(history, list):
                history = []
            
            history.append({
                't': now.strftime('%H:%M'),
                'bp': round(w5['buy_pct']),
                'p': candles[-1]['p'],
                'cvd': w5['cvd'],
            })
            
            if len(history) > 1440:
                history = history[-1440:]
            self.db.set_setting(db_key, history)
            
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
        with self._lock:
            if not self._candles:
                return {'running': self._running, 'has_data': False,
                        'scan_count': self._scan_count}
            
            candles = self._candles
            w5 = self._calc_window(candles, 5)
            w15 = self._calc_window(candles, 15)
            w60 = self._calc_window(candles, 60)
            w240 = self._calc_window(candles, 240)
            
            sig = self._signal or {}
            
            return {
                'running': self._running,
                'has_data': True,
                'price': self._price,
                'scan_count': self._scan_count,
                'errors': self._errors,
                'windows': {'5m': w5, '15m': w15, '1h': w60, '4h': w240},
                'signal': sig,
                'cvd_1h': w60['cvd'],
                'spike': w60['spike'],
                'spike_alert': w60['spike'] >= 3.0,
            }
    
    def get_history(self, date: str = '') -> Dict:
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
        return {'date': date, 'data': data, 'available_days': available}


_instance: Optional[VolumeFlow] = None

def get_volume_flow() -> Optional[VolumeFlow]:
    return _instance

def init_volume_flow(db=None, notifier=None) -> VolumeFlow:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = VolumeFlow(db=db, notifier=notifier)
    return _instance
