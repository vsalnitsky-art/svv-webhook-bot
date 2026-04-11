"""
BTC Volume Flow v3.0 — Professional Trade Signal Engine

Data sources (all Binance Futures, free, no API key):
  Every 60s:  1-min klines (taker buy/sell volumes, OHLCV)
  Every 300s: Open Interest, L/S Ratio, Top Trader L/S, Taker Ratio

Signal scoring (max 200 points):
  [50]  Multi-TF taker buy/sell dominance (5m, 15m, 1h)
  [25]  4h confirmation
  [20]  CVD trend (15min cumulative volume delta)
  [25]  Price/CVD divergence (accumulation/distribution)
  [20]  OI + Price divergence
  [20]  Long/Short ratio extreme (crowd is wrong)
  [20]  Top Trader direction (smart money)
  [20]  Taker institutional ratio

Telegram alert on direction change with confidence ≥60%.
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

SYMBOL = 'BTCUSDT'
SCAN_INTERVAL = 60
SENTIMENT_INTERVAL = 300
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
        
        self._candles: List[Dict] = []
        self._price: float = 0
        self._scan_count: int = 0
        self._errors: int = 0
        
        # Sentiment data (updated every 5min)
        self._sentiment: Dict = {}
        self._sentiment_scan: int = 0
        self._data_source: str = ''  # 'Binance', 'OKX', 'Binance+OKX'
        
        # Signal state
        self._last_signal_dir: str = ''
        self._signal: Dict = {}
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VolumeFlow")
        self._thread.start()
        print(f"[VOL FLOW] ✅ Started: {SYMBOL}, klines every {self.scan_interval}s, "
              f"sentiment every {SENTIMENT_INTERVAL}s")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _loop(self):
        print("[VOL FLOW] 🧵 Thread started")
        try:
            self._fetch_sentiment()
            self._scan()
            while self._running:
                time.sleep(self.scan_interval)
                if not self._running:
                    break
                self._scan()
                # Sentiment every 5min
                if self._scan_count % (SENTIMENT_INTERVAL // SCAN_INTERVAL) == 0:
                    self._fetch_sentiment()
        except Exception as e:
            print(f"[VOL FLOW] 💀 Crashed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # DATA FETCHING
    # ========================================
    
    def _scan(self):
        """Fetch 1-min klines with taker buy/sell volumes."""
        try:
            from detection.market_data import get_market_data
            md = get_market_data()
            
            candles = md.fetch_klines(SYMBOL, KLINE_LIMIT)
            if not candles:
                return
            
            with self._lock:
                self._candles = candles
                self._price = candles[-1]['p']
                self._scan_count += 1
                self._signal = self._calc_signal(candles)
                self._data_source = md.source_summary
            
            self._check_alert()
            self._store_snapshot(candles)
            
            if self._scan_count <= 1 or self._scan_count % 30 == 0:
                sig = self._signal
                sent = self._sentiment
                oi_str = f", OI ${sent.get('oi',0)/1e9:.1f}B" if sent.get('oi') else ""
                ls_str = f", L/S {sent.get('ls_ratio',0):.2f}" if sent.get('ls_ratio') else ""
                print(f"[VOL FLOW] #{self._scan_count}: "
                      f"{sig.get('direction','-')} {sig.get('confidence',0)}%{oi_str}{ls_str}")
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[VOL FLOW] ⚠️ Kline error #{self._errors}: {e}")
    
    def _fetch_sentiment(self):
        """Fetch OI, L/S ratio, Top Trader L/S, Taker ratio via MarketData (Binance→OKX)."""
        from detection.market_data import get_market_data
        md = get_market_data()
        sent = {}
        
        # OI
        oi_usd, oi_src = md.fetch_oi(SYMBOL, self._price)
        if oi_usd is not None:
            sent['oi'] = round(oi_usd)
            sent['oi_btc'] = round(oi_usd / self._price, 2) if self._price else 0
        
        # L/S + Top + Taker (with fallback)
        sentiment = md.fetch_sentiment(SYMBOL)
        if sentiment:
            if 'ls_ratio' in sentiment:
                sent['ls_ratio'] = sentiment['ls_ratio']
                sent['ls_long'] = sentiment.get('ls_long', 50)
                sent['ls_short'] = round(100 - sent['ls_long'], 1)
            if 'top_ls' in sentiment:
                sent['top_ls_ratio'] = sentiment['top_ls']
                sent['top_long'] = sentiment.get('top_long', 50)
                sent['top_short'] = round(100 - sent['top_long'], 1)
            if 'taker' in sentiment:
                sent['taker_ratio'] = sentiment['taker']
        
        self._sentiment_scan += 1
        with self._lock:
            prev_oi = self._sentiment.get('oi', 0)
            sent['oi_prev'] = prev_oi
            self._sentiment = sent
            self._data_source = md.source_summary
        
        if self._sentiment_scan <= 2 or self._sentiment_scan % 12 == 0:
            src = md.source_summary
            print(f"[VOL FLOW] Sentiment #{self._sentiment_scan} [{src}]: "
                  f"OI ${sent.get('oi',0)/1e9:.1f}B, "
                  f"L/S {sent.get('ls_ratio',0):.2f} ({sent.get('ls_long',0):.0f}%L), "
                  f"Top {sent.get('top_ls_ratio',0):.2f} ({sent.get('top_long',0):.0f}%L), "
                  f"Taker {sent.get('taker_ratio',0):.2f}")
    
    # ========================================
    # SIGNAL ENGINE
    # ========================================
    
    def _calc_signal(self, candles: List[Dict]) -> Dict:
        w5 = self._calc_window(candles, 5)
        w15 = self._calc_window(candles, 15)
        w60 = self._calc_window(candles, 60)
        w240 = self._calc_window(candles, 240)
        sent = self._sentiment
        
        long_score = 0
        short_score = 0
        reasons = []
        
        # === VOLUME: Multi-TF taker dominance [50+25] ===
        for label, w, pts in [('5m', w5, 18), ('15m', w15, 18), ('1h', w60, 14)]:
            if w['buy_pct'] >= 60:
                long_score += pts
                reasons.append(f"{label} Buy {w['buy_pct']:.0f}%")
            elif w['sell_pct'] >= 60:
                short_score += pts
                reasons.append(f"{label} Sell {w['sell_pct']:.0f}%")
        
        if w240['buy_pct'] >= 55:
            long_score += 15
            reasons.append(f"4h Buy {w240['buy_pct']:.0f}%")
        elif w240['sell_pct'] >= 55:
            short_score += 15
            reasons.append(f"4h Sell {w240['sell_pct']:.0f}%")
        
        # === CVD trend 15min [20] ===
        if len(candles) >= 15:
            cvd_vals = []
            running = 0
            for c in candles[-15:]:
                running += (c['b'] - c['s'])
                cvd_vals.append(running)
            if cvd_vals[-1] > cvd_vals[0] + abs(cvd_vals[0]) * 0.1:
                long_score += 20
                reasons.append("CVD rising ↑")
            elif cvd_vals[-1] < cvd_vals[0] - abs(cvd_vals[0]) * 0.1:
                short_score += 20
                reasons.append("CVD falling ↓")
        
        # === Price/CVD divergence [25] — STRONGEST ===
        if len(candles) >= 15:
            p_start = candles[-15]['p']
            p_end = candles[-1]['p']
            p_chg = (p_end - p_start) / p_start * 100
            cvd_15 = sum(c['b'] - c['s'] for c in candles[-15:])
            
            if p_chg < -0.1 and cvd_15 > 0:
                long_score += 25
                reasons.append(f"⚡ Accumulation (P{p_chg:+.2f}% CVD+)")
            elif p_chg > 0.1 and cvd_15 < 0:
                short_score += 25
                reasons.append(f"⚡ Distribution (P{p_chg:+.2f}% CVD-)")
        
        # === OI + Price divergence [20] ===
        oi = sent.get('oi', 0)
        oi_prev = sent.get('oi_prev', 0)
        if oi and oi_prev and len(candles) >= 5:
            oi_chg = (oi - oi_prev) / oi_prev * 100 if oi_prev else 0
            p_chg_5m = (candles[-1]['p'] - candles[-5]['p']) / candles[-5]['p'] * 100
            
            # OI growing + price falling = shorts accumulating → squeeze up
            if oi_chg > 0.5 and p_chg_5m < -0.1:
                long_score += 20
                reasons.append(f"OI+{oi_chg:.1f}% Price{p_chg_5m:+.1f}% → squeeze↑")
            # OI growing + price rising = longs accumulating → squeeze down
            elif oi_chg > 0.5 and p_chg_5m > 0.1:
                short_score += 20
                reasons.append(f"OI+{oi_chg:.1f}% Price{p_chg_5m:+.1f}% → squeeze↓")
            # OI dropping = liquidations
            elif oi_chg < -1.0:
                if p_chg_5m < 0:
                    short_score += 15
                    reasons.append(f"Liquidations (OI{oi_chg:+.1f}%)")
                else:
                    long_score += 15
                    reasons.append(f"Short liquidations (OI{oi_chg:+.1f}%)")
        
        # === Long/Short Ratio [20] — crowd is wrong at extremes ===
        ls = sent.get('ls_ratio', 0)
        if ls:
            # ≥2.0 = 67% long → crowd overleveraged long → SHORT signal
            if ls >= 2.0:
                short_score += 20
                reasons.append(f"L/S {ls:.2f} (crowd LONG)")
            elif ls >= 1.5:
                short_score += 10
                reasons.append(f"L/S {ls:.2f} (leaning LONG)")
            # ≤0.7 = 59% short → crowd overleveraged short → LONG signal
            elif ls <= 0.7:
                long_score += 20
                reasons.append(f"L/S {ls:.2f} (crowd SHORT)")
            elif ls <= 0.85:
                long_score += 10
                reasons.append(f"L/S {ls:.2f} (leaning SHORT)")
        
        # === Top Trader direction [20] — smart money ===
        top_ls = sent.get('top_ls_ratio', 0)
        if top_ls:
            if top_ls >= 1.3:
                long_score += 20
                reasons.append(f"Top traders LONG ({sent.get('top_long',0):.0f}%)")
            elif top_ls >= 1.1:
                long_score += 10
                reasons.append(f"Top traders lean LONG ({sent.get('top_long',0):.0f}%)")
            elif top_ls <= 0.75:
                short_score += 20
                reasons.append(f"Top traders SHORT ({sent.get('top_short',0):.0f}%)")
            elif top_ls <= 0.9:
                short_score += 10
                reasons.append(f"Top traders lean SHORT ({sent.get('top_short',0):.0f}%)")
        
        # === Taker institutional ratio [20] ===
        taker = sent.get('taker_ratio', 0)
        if taker:
            if taker >= 1.15:
                long_score += 20
                reasons.append(f"Taker Buy dominant ({taker:.2f})")
            elif taker >= 1.05:
                long_score += 10
                reasons.append(f"Taker Buy lean ({taker:.2f})")
            elif taker <= 0.85:
                short_score += 20
                reasons.append(f"Taker Sell dominant ({taker:.2f})")
            elif taker <= 0.95:
                short_score += 10
                reasons.append(f"Taker Sell lean ({taker:.2f})")
        
        # === Volume spike bonus [10] ===
        spike = w60['spike']
        if spike >= 2.0:
            last5 = candles[-5:]
            sb = sum(c['b'] for c in last5)
            ss = sum(c['s'] for c in last5)
            if sb > ss * 1.3:
                long_score += 10
                reasons.append(f"Spike {spike:.1f}× Buy")
            elif ss > sb * 1.3:
                short_score += 10
                reasons.append(f"Spike {spike:.1f}× Sell")
        
        # === Calculate final signal ===
        total = long_score + short_score
        if total == 0:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': [],
                    'long_score': 0, 'short_score': 0, 'price': self._price}
        
        if long_score > short_score:
            confidence = min(95, round(long_score / 2.0))  # /200 * 100
            direction = 'LONG'
        elif short_score > long_score:
            confidence = min(95, round(short_score / 2.0))
            direction = 'SHORT'
        else:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': reasons,
                    'long_score': long_score, 'short_score': short_score, 'price': self._price}
        
        return {
            'direction': direction,
            'confidence': confidence,
            'long_score': long_score,
            'short_score': short_score,
            'reasons': reasons,
            'price': self._price,
        }
    
    def _check_alert(self):
        sig = self._signal
        if not sig or sig.get('confidence', 0) < 60:
            return
        direction = sig.get('direction', '')
        if not direction or direction == 'NEUTRAL' or direction == self._last_signal_dir:
            return
        self._last_signal_dir = direction
        if not self.notifier:
            print(f"[VOL FLOW] 🔔 {direction} {sig['confidence']}% (no TG)")
            return
        try:
            icon = '🟢' if direction == 'LONG' else '🔴'
            reasons_str = '\n'.join(f"  • {r}" for r in sig.get('reasons', []))
            msg = (
                f"{icon} <b>BTC SIGNAL: {direction} {sig['confidence']}%</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 BTC ${self._price:,.2f}\n\n"
                f"📊 <b>Factors:</b>\n{reasons_str}\n\n"
                f"L: {sig.get('long_score',0)} | S: {sig.get('short_score',0)}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⏱ {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
            )
            self.notifier.send_message(msg)
            print(f"[VOL FLOW] 📨 TG: {direction} {sig['confidence']}%")
        except Exception as e:
            print(f"[VOL FLOW] ⚠️ Alert error: {e}")
    
    # ========================================
    # HELPERS
    # ========================================
    
    def _calc_window(self, candles: List[Dict], minutes: int) -> Dict:
        recent = candles[-minutes:] if len(candles) >= minutes else candles
        tb = sum(c['b'] for c in recent)
        ts = sum(c['s'] for c in recent)
        t = tb + ts
        bp = (tb / t * 100) if t > 0 else 50
        avg = t / len(recent) if recent else 0
        last = recent[-1]['v'] if recent else 0
        spike = (last / avg) if avg > 0 else 1
        sig = 'BUYERS' if bp >= 60 else ('SELLERS' if bp <= 40 else 'NEUTRAL')
        return {
            'buy': round(tb), 'sell': round(ts), 'total': round(t),
            'buy_pct': round(bp, 1), 'sell_pct': round(100-bp, 1),
            'cvd': round(tb - ts), 'signal': sig,
            'spike': round(spike, 1), 'avg_vol_min': round(avg),
        }
    
    def _store_snapshot(self, candles):
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
        except:
            pass
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_summary(self) -> Dict:
        with self._lock:
            if not self._candles:
                return {'running': self._running, 'has_data': False,
                        'scan_count': self._scan_count}
            c = self._candles
            sent = self._sentiment
            return {
                'running': self._running, 'has_data': True,
                'price': self._price,
                'scan_count': self._scan_count, 'errors': self._errors,
                'windows': {
                    '5m': self._calc_window(c, 5),
                    '15m': self._calc_window(c, 15),
                    '1h': self._calc_window(c, 60),
                    '4h': self._calc_window(c, 240),
                },
                'signal': self._signal or {},
                'sentiment': {
                    'oi': sent.get('oi', 0),
                    'oi_btc': sent.get('oi_btc', 0),
                    'ls_ratio': sent.get('ls_ratio', 0),
                    'ls_long': sent.get('ls_long', 0),
                    'top_ls_ratio': sent.get('top_ls_ratio', 0),
                    'top_long': sent.get('top_long', 0),
                    'taker_ratio': sent.get('taker_ratio', 0),
                },
                'cvd_1h': self._calc_window(c, 60)['cvd'],
                'spike': self._calc_window(c, 60)['spike'],
                'spike_alert': self._calc_window(c, 60)['spike'] >= 3.0,
                'data_source': self._data_source or 'Binance',
            }
    
    def get_history(self, date=''):
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
def get_volume_flow(): return _instance
def init_volume_flow(db=None, notifier=None):
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = VolumeFlow(db=db, notifier=notifier)
    return _instance
