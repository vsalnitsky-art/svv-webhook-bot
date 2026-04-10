"""
Coin Flow Scanner v1.0 — Volume Flow analysis for Funding watchlist coins

For each coin in Funding watchlist:
  - Klines 1m (taker buy/sell) every 5min
  - OI every 5min
  - L/S Ratio, Top Trader, Taker Ratio every 15min

Sequential requests with 0.5s delays to avoid rate limits.
"""

import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

BINANCE_KLINE_URL = 'https://fapi.binance.com/fapi/v1/klines'
BINANCE_OI_URL = 'https://fapi.binance.com/fapi/v1/openInterest'
BINANCE_LS_URL = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
BINANCE_TOP_LS_URL = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'
BINANCE_TAKER_URL = 'https://fapi.binance.com/futures/data/takerlongshortRatio'

SCAN_INTERVAL = 300       # Klines + OI every 5 min
SENTIMENT_EVERY = 3       # Sentiment every 3rd scan = 15 min
REQUEST_DELAY = 0.5       # 500ms between requests
KLINE_LIMIT = 60          # 1 hour of 1-min candles per coin


class CoinFlowScanner:
    
    def __init__(self, funding_monitor=None, notifier=None):
        self.funding_monitor = funding_monitor
        self.notifier = notifier
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
        
        # {symbol: {kline_data, oi, sentiment, signal, last_update}}
        self._data: Dict[str, Dict] = {}
        self._scan_count: int = 0
        self._errors: int = 0
        
        # Track alerts per coin (one per direction change)
        self._last_alerts: Dict[str, str] = {}  # symbol -> last direction
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CoinFlow")
        self._thread.start()
        print(f"[COIN FLOW] ✅ Started: klines+OI every {SCAN_INTERVAL}s, "
              f"sentiment every {SCAN_INTERVAL * SENTIMENT_EVERY}s")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
    
    def _loop(self):
        print("[COIN FLOW] 🧵 Thread started")
        time.sleep(30)  # Wait for funding monitor to populate
        try:
            while self._running:
                self._scan()
                for _ in range(SCAN_INTERVAL):
                    if not self._running:
                        return
                    time.sleep(1)
        except Exception as e:
            print(f"[COIN FLOW] 💀 Crashed: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_watchlist_symbols(self) -> List[str]:
        if not self.funding_monitor:
            return []
        wl = self.funding_monitor.get_watchlist()
        return [c['symbol'] for c in wl.get('coins', [])]
    
    def _scan(self):
        symbols = self._get_watchlist_symbols()
        if not symbols:
            return
        
        self._scan_count += 1
        do_sentiment = (self._scan_count % SENTIMENT_EVERY == 0)
        now_str = datetime.now(timezone.utc).strftime('%H:%M')
        
        for symbol in symbols:
            if not self._running:
                return
            try:
                coin_data = self._data.get(symbol, {})
                
                # 1. Klines (always)
                klines = self._fetch_klines(symbol)
                if klines:
                    coin_data['klines'] = klines
                time.sleep(REQUEST_DELAY)
                
                # 2. OI (always)
                oi = self._fetch_oi(symbol)
                if oi is not None:
                    prev_oi = coin_data.get('oi', oi)
                    coin_data['oi_prev'] = prev_oi
                    coin_data['oi'] = oi
                time.sleep(REQUEST_DELAY)
                
                # 3. Sentiment (every 15min)
                if do_sentiment:
                    sent = self._fetch_sentiment(symbol)
                    if sent:
                        coin_data['sentiment'] = sent
                    time.sleep(REQUEST_DELAY)
                
                # 4. Calculate signal
                coin_data['last_update'] = now_str
                signal = self._calc_signal(symbol, coin_data)
                coin_data['signal'] = signal
                
                with self._lock:
                    self._data[symbol] = coin_data
                
                # 5. Check alert
                self._check_alert(symbol, signal)
                
            except Exception as e:
                self._errors += 1
                if self._errors <= 5:
                    print(f"[COIN FLOW] ⚠️ {symbol}: {e}")
        
        # Clean removed coins
        with self._lock:
            for sym in list(self._data.keys()):
                if sym not in symbols:
                    del self._data[sym]
        
        if self._scan_count <= 2 or self._scan_count % 12 == 0:
            print(f"[COIN FLOW] #{self._scan_count}: {len(symbols)} coins"
                  f"{' +sentiment' if do_sentiment else ''}")
    
    # ========================================
    # DATA FETCHING
    # ========================================
    
    def _fetch_klines(self, symbol: str) -> Optional[List[Dict]]:
        try:
            r = self._session.get(BINANCE_KLINE_URL,
                params={'symbol': symbol, 'interval': '1m', 'limit': KLINE_LIMIT},
                timeout=10)
            if r.status_code != 200:
                return None
            raw = r.json()
            candles = []
            for k in raw:
                try:
                    tv = float(k[7])
                    tb = float(k[10])
                    candles.append({
                        'p': float(k[4]), 'v': round(tv),
                        'b': round(tb), 's': round(tv - tb),
                    })
                except:
                    continue
            return candles if candles else None
        except:
            return None
    
    def _fetch_oi(self, symbol: str) -> Optional[float]:
        try:
            r = self._session.get(BINANCE_OI_URL,
                params={'symbol': symbol}, timeout=10)
            if r.status_code == 200:
                d = r.json()
                oi_qty = float(d.get('openInterest', 0))
                price = self._data.get(symbol, {}).get('klines', [{}])
                last_p = price[-1]['p'] if price else 0
                return oi_qty * last_p if last_p else oi_qty
        except:
            pass
        return None
    
    def _fetch_sentiment(self, symbol: str) -> Optional[Dict]:
        sent = {}
        try:
            r = self._session.get(BINANCE_LS_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    sent['ls_ratio'] = float(latest.get('longShortRatio', 1))
                    sent['ls_long'] = round(float(latest.get('longAccount', 0.5)) * 100, 1)
        except:
            pass
        time.sleep(REQUEST_DELAY)
        
        try:
            r = self._session.get(BINANCE_TOP_LS_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    sent['top_ls'] = float(latest.get('longShortRatio', 1))
                    sent['top_long'] = round(float(latest.get('longAccount', 0.5)) * 100, 1)
        except:
            pass
        time.sleep(REQUEST_DELAY)
        
        try:
            r = self._session.get(BINANCE_TAKER_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    sent['taker'] = float(data[-1].get('buySellRatio', 1))
        except:
            pass
        
        return sent if sent else None
    
    # ========================================
    # SIGNAL ENGINE (same logic as BTC Volume Flow)
    # ========================================
    
    def _calc_window(self, candles: List[Dict], minutes: int) -> Dict:
        recent = candles[-minutes:] if len(candles) >= minutes else candles
        if not recent:
            return {'buy_pct': 50, 'sell_pct': 50, 'cvd': 0, 'signal': 'NEUTRAL',
                    'buy': 0, 'sell': 0, 'total': 0, 'spike': 1}
        tb = sum(c['b'] for c in recent)
        ts = sum(c['s'] for c in recent)
        t = tb + ts
        bp = (tb / t * 100) if t > 0 else 50
        avg = t / len(recent) if recent else 1
        last = recent[-1]['v'] if recent else 0
        spike = (last / avg) if avg > 0 else 1
        sig = 'BUYERS' if bp >= 60 else ('SELLERS' if bp <= 40 else 'NEUTRAL')
        return {
            'buy': round(tb), 'sell': round(ts), 'total': round(t),
            'buy_pct': round(bp, 1), 'sell_pct': round(100 - bp, 1),
            'cvd': round(tb - ts), 'signal': sig, 'spike': round(spike, 1),
        }
    
    def _calc_signal(self, symbol: str, coin: Dict) -> Dict:
        klines = coin.get('klines', [])
        if len(klines) < 5:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': []}
        
        sent = coin.get('sentiment', {})
        w5 = self._calc_window(klines, 5)
        w15 = self._calc_window(klines, 15)
        w60 = self._calc_window(klines, 60)
        
        long_s = 0
        short_s = 0
        reasons = []
        
        # Volume dominance [50]
        for label, w, pts in [('5m', w5, 18), ('15m', w15, 18), ('1h', w60, 14)]:
            if w['buy_pct'] >= 60:
                long_s += pts; reasons.append(f"{label} Buy {w['buy_pct']:.0f}%")
            elif w['sell_pct'] >= 60:
                short_s += pts; reasons.append(f"{label} Sell {w['sell_pct']:.0f}%")
        
        # CVD trend [20]
        if len(klines) >= 15:
            cvd_vals = []
            run = 0
            for c in klines[-15:]:
                run += (c['b'] - c['s'])
                cvd_vals.append(run)
            if cvd_vals[-1] > cvd_vals[0] + abs(cvd_vals[0]) * 0.1:
                long_s += 20; reasons.append("CVD rising ↑")
            elif cvd_vals[-1] < cvd_vals[0] - abs(cvd_vals[0]) * 0.1:
                short_s += 20; reasons.append("CVD falling ↓")
        
        # Divergence [25]
        if len(klines) >= 15:
            pc = (klines[-1]['p'] - klines[-15]['p']) / klines[-15]['p'] * 100
            cvd15 = sum(c['b'] - c['s'] for c in klines[-15:])
            if pc < -0.1 and cvd15 > 0:
                long_s += 25; reasons.append(f"⚡ Accumulation (P{pc:+.2f}%)")
            elif pc > 0.1 and cvd15 < 0:
                short_s += 25; reasons.append(f"⚡ Distribution (P{pc:+.2f}%)")
        
        # OI divergence [20]
        oi = coin.get('oi', 0)
        oi_prev = coin.get('oi_prev', 0)
        if oi and oi_prev and len(klines) >= 5:
            oi_chg = (oi - oi_prev) / oi_prev * 100 if oi_prev else 0
            pc5 = (klines[-1]['p'] - klines[-5]['p']) / klines[-5]['p'] * 100
            if oi_chg > 0.5 and pc5 < -0.1:
                long_s += 20; reasons.append(f"OI↑ Price↓ squeeze↑")
            elif oi_chg > 0.5 and pc5 > 0.1:
                short_s += 20; reasons.append(f"OI↑ Price↑ squeeze↓")
        
        # L/S ratio [20]
        ls = sent.get('ls_ratio', 0)
        if ls >= 2.0:
            short_s += 20; reasons.append(f"L/S {ls:.2f} crowd LONG")
        elif ls <= 0.7:
            long_s += 20; reasons.append(f"L/S {ls:.2f} crowd SHORT")
        elif ls >= 1.5:
            short_s += 10; reasons.append(f"L/S {ls:.2f}")
        elif ls <= 0.85:
            long_s += 10; reasons.append(f"L/S {ls:.2f}")
        
        # Top trader [20]
        top = sent.get('top_ls', 0)
        if top >= 1.3:
            long_s += 20; reasons.append(f"Top LONG {sent.get('top_long',0):.0f}%")
        elif top <= 0.75:
            short_s += 20; reasons.append(f"Top SHORT")
        elif top >= 1.1:
            long_s += 10; reasons.append(f"Top lean LONG")
        elif top <= 0.9:
            short_s += 10; reasons.append(f"Top lean SHORT")
        
        # Taker [15]
        taker = sent.get('taker', 0)
        if taker >= 1.15:
            long_s += 15; reasons.append(f"Taker Buy {taker:.2f}")
        elif taker <= 0.85:
            short_s += 15; reasons.append(f"Taker Sell {taker:.2f}")
        
        total = long_s + short_s
        if total == 0:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': [],
                    'long_score': 0, 'short_score': 0}
        
        if long_s > short_s:
            conf = min(95, round(long_s / 1.7))
            direction = 'LONG'
        elif short_s > long_s:
            conf = min(95, round(short_s / 1.7))
            direction = 'SHORT'
        else:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': reasons,
                    'long_score': long_s, 'short_score': short_s}
        
        return {
            'direction': direction, 'confidence': conf,
            'long_score': long_s, 'short_score': short_s,
            'reasons': reasons,
        }
    
    def _check_alert(self, symbol: str, signal: Dict):
        if not signal or signal.get('confidence', 0) < 65:
            return
        direction = signal.get('direction', '')
        if not direction or direction == 'NEUTRAL':
            return
        if self._last_alerts.get(symbol) == direction:
            return
        self._last_alerts[symbol] = direction
        
        if not self.notifier:
            print(f"[COIN FLOW] 🔔 {symbol}: {direction} {signal['confidence']}%")
            return
        try:
            icon = '🟢' if direction == 'LONG' else '🔴'
            reasons_str = '\n'.join(f"  • {r}" for r in signal.get('reasons', []))
            msg = (
                f"{icon} <b>{symbol} SIGNAL: {direction} {signal['confidence']}%</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Factors:</b>\n{reasons_str}\n\n"
                f"L: {signal.get('long_score',0)} | S: {signal.get('short_score',0)}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⏱ {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
            )
            self.notifier.send_message(msg)
            print(f"[COIN FLOW] 📨 {symbol}: {direction} {signal['confidence']}%")
        except Exception as e:
            print(f"[COIN FLOW] ⚠️ Alert error {symbol}: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_coin_summary(self, symbol: str) -> Dict:
        with self._lock:
            coin = self._data.get(symbol, {})
            if not coin:
                return {'found': False, 'symbol': symbol}
            
            klines = coin.get('klines', [])
            sent = coin.get('sentiment', {})
            sig = coin.get('signal', {})
            
            result = {
                'found': True, 'symbol': symbol,
                'signal': sig,
                'last_update': coin.get('last_update', ''),
                'sentiment': sent,
                'oi': coin.get('oi', 0),
            }
            
            if klines:
                result['windows'] = {
                    '5m': self._calc_window(klines, 5),
                    '15m': self._calc_window(klines, 15),
                    '1h': self._calc_window(klines, 60),
                }
                result['price'] = klines[-1]['p']
                result['cvd_15m'] = sum(c['b'] - c['s'] for c in klines[-15:]) if len(klines) >= 15 else 0
            
            return result
    
    def get_all_signals(self) -> Dict:
        """Summary signals for all tracked coins."""
        with self._lock:
            signals = {}
            for sym, coin in self._data.items():
                sig = coin.get('signal', {})
                signals[sym] = {
                    'direction': sig.get('direction', 'NEUTRAL'),
                    'confidence': sig.get('confidence', 0),
                    'reasons_count': len(sig.get('reasons', [])),
                    'last_update': coin.get('last_update', ''),
                }
            return {
                'signals': signals,
                'scan_count': self._scan_count,
                'total': len(self._data),
                'running': self._running,
            }


_instance: Optional[CoinFlowScanner] = None
def get_coin_flow() -> Optional[CoinFlowScanner]: return _instance
def init_coin_flow(funding_monitor=None, notifier=None) -> CoinFlowScanner:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = CoinFlowScanner(funding_monitor=funding_monitor, notifier=notifier)
    return _instance
