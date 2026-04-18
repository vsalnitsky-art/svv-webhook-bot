"""
Position Exit Monitor v1.0 — Intelligent Exit Signal Analysis

For each open trade, calculates Exit Score (0-100) based on 8 exhaustion factors:
  1. CVD Divergence (25 pts) — price up, CVD down = buyers exhausted
  2. Volume Flow Flip (20 pts) — entry direction signal lost
  3. OI Divergence (15 pts) — OI falling while price rising = closing, not opening
  4. Top Trader Flip (15 pts) — smart money switched sides
  5. L/S Ratio Extreme (10 pts) — crowd crowded on wrong side
  6. Approach to Wall (5 pts) — liquidity wall ahead blocks further move
  7. RSI Divergence (5 pts) — classical momentum divergence
  8. Funding Extremum (5 pts) — funding overheated

Thresholds (configurable via settings):
  - 0-30:   HEALTHY — hold position
  - 30-50:  WARNING — tighten trailing stop
  - 50-70:  PARTIAL EXIT — close 50%
  - 70-100: FULL EXIT — close entire position immediately

Data sources:
  - Position list: 'internal' (db.get_open_trades) or 'bybit' (get_positions)
  - Market data: MarketData (Binance→OKX→Bybit fallback)

Currently: Telegram alerts only. Auto-close disabled by default (future toggle).
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

SCAN_INTERVAL = 10        # seconds — as requested
DB_KEY_HISTORY = 'exit_monitor_history'
DB_KEY_SETTINGS = 'exit_monitor_settings'
HISTORY_DAYS = 5


# Default factor weights (configurable in settings)
DEFAULT_WEIGHTS = {
    'cvd_divergence':  25,
    'volflow_flip':    20,
    'oi_divergence':   15,
    'top_trader_flip': 15,
    'ls_extreme':      10,
    'wall_approach':    5,
    'rsi_divergence':   5,
    'funding_extreme':  5,
}

# Default thresholds
DEFAULT_THRESHOLDS = {
    'warning':      30,
    'partial_exit': 50,
    'full_exit':    70,
}

# Default settings
DEFAULT_SETTINGS = {
    'enabled': True,
    'data_source': 'internal',   # 'internal' | 'bybit'
    'auto_close': False,         # future toggle
    'telegram_alerts': True,
    'scan_interval': SCAN_INTERVAL,
    'weights': DEFAULT_WEIGHTS.copy(),
    'thresholds': DEFAULT_THRESHOLDS.copy(),
}


class PositionExitMonitor:
    
    def __init__(self, db=None, notifier=None, bybit_connector=None):
        self.db = db
        self.notifier = notifier
        self.bybit = bybit_connector
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Current state per symbol: {symbol: {score, factors, action, price, ...}}
        self._state: Dict[str, Dict] = {}
        
        # Alerts sent tracking: {symbol: last_action_level} to avoid spam
        self._last_alerts: Dict[str, str] = {}
        
        # History for charts: {date: [{t, symbol, score, action}, ...]}
        self._scan_count = 0
        self._errors = 0
        
        self._settings = self._load_settings()
    
    # ========================================
    # SETTINGS
    # ========================================
    
    def _load_settings(self) -> Dict:
        if not self.db:
            return DEFAULT_SETTINGS.copy()
        try:
            stored = self.db.get_setting(DB_KEY_SETTINGS, None)
            if isinstance(stored, dict):
                # Merge with defaults (to add new keys)
                settings = DEFAULT_SETTINGS.copy()
                settings.update(stored)
                # Merge nested weights
                w = DEFAULT_WEIGHTS.copy()
                w.update(stored.get('weights', {}))
                settings['weights'] = w
                t = DEFAULT_THRESHOLDS.copy()
                t.update(stored.get('thresholds', {}))
                settings['thresholds'] = t
                return settings
        except:
            pass
        return DEFAULT_SETTINGS.copy()
    
    def get_settings(self) -> Dict:
        with self._lock:
            return dict(self._settings)
    
    def update_settings(self, new_settings: Dict) -> bool:
        try:
            with self._lock:
                self._settings.update(new_settings)
                if self.db:
                    self.db.set_setting(DB_KEY_SETTINGS, self._settings)
            return True
        except Exception as e:
            print(f"[EXIT] Settings error: {e}")
            return False
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self):
        if self._running:
            return
        if not self._settings.get('enabled', True):
            print("[EXIT] Disabled in settings, not starting")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ExitMonitor")
        self._thread.start()
        interval = self._settings.get('scan_interval', SCAN_INTERVAL)
        print(f"[EXIT] ✅ Started: scanning every {interval}s, source={self._settings.get('data_source')}")
    
    def stop(self):
        self._running = False
    
    def _loop(self):
        print("[EXIT] 🧵 Thread started")
        time.sleep(10)  # Wait for other modules to initialize
        try:
            while self._running:
                try:
                    self._scan()
                except Exception as e:
                    self._errors += 1
                    if self._errors <= 5:
                        print(f"[EXIT] Scan error: {e}")
                
                interval = self._settings.get('scan_interval', SCAN_INTERVAL)
                for _ in range(interval):
                    if not self._running:
                        return
                    time.sleep(1)
        except Exception as e:
            print(f"[EXIT] 💀 Crashed: {e}")
    
    # ========================================
    # POSITION FETCHING
    # ========================================
    
    def _get_open_positions(self) -> List[Dict]:
        """Get open positions from configured source."""
        source = self._settings.get('data_source', 'internal')
        
        if source == 'bybit':
            return self._get_bybit_positions()
        else:
            return self._get_internal_trades()
    
    def _get_internal_trades(self) -> List[Dict]:
        """Get open trades from internal DB."""
        if not self.db:
            return []
        try:
            trades = self.db.get_open_trades() or []
            result = []
            for t in trades:
                if not isinstance(t, dict):
                    t = dict(t) if hasattr(t, '__dict__') else {}
                result.append({
                    'symbol': t.get('symbol', ''),
                    'side': t.get('side', '').upper(),
                    'entry_price': float(t.get('entry_price', 0) or 0),
                    'qty': float(t.get('qty', 0) or 0),
                    'sl': float(t.get('sl_price', t.get('stop_loss', 0)) or 0),
                    'tp': float(t.get('tp_price', t.get('take_profit', 0)) or 0),
                    'opened_at': t.get('created_at', t.get('opened_at', '')),
                    'id': t.get('id', ''),
                    'source': 'internal',
                })
            return result
        except Exception as e:
            print(f"[EXIT] Internal trades error: {e}")
            return []
    
    def _get_bybit_positions(self) -> List[Dict]:
        """Get open positions from Bybit API."""
        if not self.bybit:
            return []
        try:
            session = getattr(self.bybit, 'session', None) or self.bybit
            resp = session.get_positions(category='linear', settleCoin='USDT')
            if resp.get('retCode') != 0:
                return []
            result = []
            for p in resp.get('result', {}).get('list', []):
                if float(p.get('size', 0)) <= 0:
                    continue
                bybit_side = p.get('side', '')  # 'Buy' or 'Sell'
                side = 'LONG' if bybit_side == 'Buy' else 'SHORT'
                result.append({
                    'symbol': p.get('symbol', ''),
                    'side': side,
                    'entry_price': float(p.get('avgPrice', 0) or 0),
                    'qty': float(p.get('size', 0) or 0),
                    'sl': float(p.get('stopLoss', 0) or 0),
                    'tp': float(p.get('takeProfit', 0) or 0),
                    'opened_at': p.get('createdTime', ''),
                    'id': p.get('symbol', ''),
                    'source': 'bybit',
                })
            return result
        except Exception as e:
            print(f"[EXIT] Bybit positions error: {e}")
            return []
    
    # ========================================
    # SCAN
    # ========================================
    
    def _scan(self):
        positions = self._get_open_positions()
        self._scan_count += 1
        
        if not positions:
            with self._lock:
                self._state = {}
            return
        
        from detection.market_data import get_market_data
        md = get_market_data()
        
        current_symbols = set()
        
        for pos in positions:
            symbol = pos['symbol']
            if not symbol:
                continue
            current_symbols.add(symbol)
            
            try:
                analysis = self._analyze_position(pos, md)
                with self._lock:
                    self._state[symbol] = analysis
                
                self._check_alert(symbol, pos, analysis)
                
                # Store history every 6 scans (1 min at 10s interval)
                if self._scan_count % 6 == 0:
                    self._store_history(symbol, analysis)
            except Exception as e:
                if self._errors <= 5:
                    print(f"[EXIT] Analyze error {symbol}: {e}")
        
        # Clean state for closed positions
        with self._lock:
            for sym in list(self._state.keys()):
                if sym not in current_symbols:
                    del self._state[sym]
                    if sym in self._last_alerts:
                        del self._last_alerts[sym]
        
        if self._scan_count <= 2 or self._scan_count % 60 == 0:
            print(f"[EXIT] Scan #{self._scan_count}: {len(positions)} positions tracked")
    
    # ========================================
    # ANALYSIS (8 FACTORS)
    # ========================================
    
    def _analyze_position(self, pos: Dict, md) -> Dict:
        symbol = pos['symbol']
        side = pos['side']
        entry = pos['entry_price']
        is_long = side == 'LONG'
        
        # Fetch all needed data
        klines = md.fetch_klines(symbol, 60)
        oi_usd, _ = md.fetch_oi(symbol, klines[-1]['p'] if klines else 0)
        sentiment = md.fetch_sentiment(symbol)
        
        if not klines or len(klines) < 20:
            return self._empty_analysis(pos, 'No data')
        
        current_price = klines[-1]['p']
        pnl_pct = ((current_price - entry) / entry * 100) if entry else 0
        if not is_long:
            pnl_pct = -pnl_pct
        
        weights = self._settings.get('weights', DEFAULT_WEIGHTS)
        factors = {}
        reasons = []
        score = 0
        
        # === Factor 1: CVD Divergence (25 pts) ===
        cvd_score = self._factor_cvd_divergence(klines, is_long)
        factors['cvd_divergence'] = cvd_score
        if cvd_score > 0:
            pts = round(cvd_score * weights.get('cvd_divergence', 25) / 100)
            score += pts
            if pts >= 10:
                reasons.append(f"CVD divergence {pts}p")
        
        # === Factor 2: Volume Flow Flip (20 pts) ===
        vf_score, vf_sig = self._factor_volflow_flip(klines, sentiment, is_long)
        factors['volflow_flip'] = vf_score
        if vf_score > 0:
            pts = round(vf_score * weights.get('volflow_flip', 20) / 100)
            score += pts
            if pts >= 8:
                reasons.append(f"Vol flipped to {vf_sig} {pts}p")
        
        # === Factor 3: OI Divergence (15 pts) ===
        oi_score = self._factor_oi_divergence(klines, oi_usd, is_long, symbol)
        factors['oi_divergence'] = oi_score
        if oi_score > 0:
            pts = round(oi_score * weights.get('oi_divergence', 15) / 100)
            score += pts
            if pts >= 6:
                reasons.append(f"OI diverging {pts}p")
        
        # === Factor 4: Top Trader Flip (15 pts) ===
        top_score = self._factor_top_trader(sentiment, is_long)
        factors['top_trader_flip'] = top_score
        if top_score > 0:
            pts = round(top_score * weights.get('top_trader_flip', 15) / 100)
            score += pts
            if pts >= 6:
                reasons.append(f"Top trader flip {pts}p")
        
        # === Factor 5: L/S Extreme (10 pts) ===
        ls_score = self._factor_ls_extreme(sentiment, is_long)
        factors['ls_extreme'] = ls_score
        if ls_score > 0:
            pts = round(ls_score * weights.get('ls_extreme', 10) / 100)
            score += pts
            if pts >= 4:
                reasons.append(f"Crowd extreme {pts}p")
        
        # === Factor 6: Wall Approach (5 pts) ===
        wall_score = self._factor_wall_approach(symbol, current_price, is_long)
        factors['wall_approach'] = wall_score
        if wall_score > 0:
            pts = round(wall_score * weights.get('wall_approach', 5) / 100)
            score += pts
            if pts >= 2:
                reasons.append(f"Wall ahead {pts}p")
        
        # === Factor 7: RSI Divergence (5 pts) ===
        rsi_score = self._factor_rsi_divergence(klines, is_long)
        factors['rsi_divergence'] = rsi_score
        if rsi_score > 0:
            pts = round(rsi_score * weights.get('rsi_divergence', 5) / 100)
            score += pts
            if pts >= 2:
                reasons.append(f"RSI div {pts}p")
        
        # === Factor 8: Funding Extreme (5 pts) ===
        fund_score = self._factor_funding_extreme(symbol, is_long)
        factors['funding_extreme'] = fund_score
        if fund_score > 0:
            pts = round(fund_score * weights.get('funding_extreme', 5) / 100)
            score += pts
            if pts >= 2:
                reasons.append(f"Funding extreme {pts}p")
        
        # Cap at 100
        score = min(100, score)
        
        # Determine action
        th = self._settings.get('thresholds', DEFAULT_THRESHOLDS)
        if score >= th.get('full_exit', 70):
            action = 'FULL_EXIT'
            action_text = '🚨 CLOSE NOW'
        elif score >= th.get('partial_exit', 50):
            action = 'PARTIAL_EXIT'
            action_text = '⚠️ TRIM 50%'
        elif score >= th.get('warning', 30):
            action = 'WARNING'
            action_text = '⚠️ TIGHTEN SL'
        else:
            action = 'HOLD'
            action_text = '✅ HOLD'
        
        return {
            'symbol': symbol,
            'side': side,
            'entry': entry,
            'current_price': current_price,
            'pnl_pct': round(pnl_pct, 2),
            'qty': pos.get('qty', 0),
            'score': score,
            'action': action,
            'action_text': action_text,
            'factors': factors,
            'reasons': reasons,
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def _empty_analysis(self, pos, reason):
        return {
            'symbol': pos['symbol'],
            'side': pos['side'],
            'entry': pos['entry_price'],
            'current_price': 0,
            'pnl_pct': 0,
            'qty': pos.get('qty', 0),
            'score': 0,
            'action': 'HOLD',
            'action_text': f'— ({reason})',
            'factors': {},
            'reasons': [],
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    # ========================================
    # FACTOR CALCULATIONS (each returns 0-100)
    # ========================================
    
    def _factor_cvd_divergence(self, klines, is_long):
        """Price rising but CVD falling (for LONG) = exhaustion."""
        if len(klines) < 15:
            return 0
        
        recent = klines[-15:]
        price_chg = (recent[-1]['p'] - recent[0]['p']) / recent[0]['p'] * 100 if recent[0]['p'] else 0
        cvd_total = sum(c.get('b', 0) - c.get('s', 0) for c in recent)
        
        if is_long:
            # Price up, CVD down = divergence
            if price_chg > 0.1 and cvd_total < 0:
                # Stronger divergence = higher score
                magnitude = min(100, int(abs(cvd_total / recent[-1]['v'] * 100)) if recent[-1].get('v', 0) > 0 else 50)
                return min(100, 60 + magnitude)
            if price_chg > 0.1 and cvd_total < recent[0].get('b', 0) * 0.3:
                return 40
        else:
            # Price down, CVD up = SHORT exhaustion
            if price_chg < -0.1 and cvd_total > 0:
                magnitude = min(100, int(abs(cvd_total / recent[-1]['v'] * 100)) if recent[-1].get('v', 0) > 0 else 50)
                return min(100, 60 + magnitude)
            if price_chg < -0.1 and cvd_total > abs(recent[0].get('s', 0)) * 0.3:
                return 40
        return 0
    
    def _factor_volflow_flip(self, klines, sentiment, is_long):
        """Current Volume Flow signal opposite to position direction."""
        if len(klines) < 15:
            return 0, 'NEUTRAL'
        
        recent5 = klines[-5:]
        recent15 = klines[-15:]
        
        def pct(candles):
            tb = sum(c.get('b', 0) for c in candles)
            ts = sum(c.get('s', 0) for c in candles)
            t = tb + ts
            return (tb / t * 100) if t > 0 else 50
        
        p5 = pct(recent5)
        p15 = pct(recent15)
        
        # For LONG: current signal shows SELLERS dominant
        if is_long:
            if p5 <= 40 and p15 <= 45:
                return 90, 'SHORT'
            if p5 <= 45:
                return 50, 'WEAK'
        else:
            if p5 >= 60 and p15 >= 55:
                return 90, 'LONG'
            if p5 >= 55:
                return 50, 'WEAK'
        return 0, 'NEUTRAL'
    
    def _factor_oi_divergence(self, klines, oi_usd, is_long, symbol):
        """OI falling while price in favor = position closing, not building."""
        # Need prev OI to compare — store it
        prev_oi = self._state.get(symbol, {}).get('_prev_oi', 0)
        
        if not prev_oi or not oi_usd:
            # Store for next scan
            self._state.setdefault(symbol, {})['_prev_oi'] = oi_usd
            return 0
        
        oi_chg = (oi_usd - prev_oi) / prev_oi * 100 if prev_oi else 0
        
        # Update for next scan
        self._state.setdefault(symbol, {})['_prev_oi'] = oi_usd
        
        if len(klines) < 5:
            return 0
        
        price_chg = (klines[-1]['p'] - klines[-5]['p']) / klines[-5]['p'] * 100 if klines[-5]['p'] else 0
        
        if is_long:
            # Price up, OI down = traders closing LONG
            if price_chg > 0.1 and oi_chg < -0.3:
                return 90
            if price_chg > 0.05 and oi_chg < -0.1:
                return 50
        else:
            # Price down, OI down = traders closing SHORT
            if price_chg < -0.1 and oi_chg < -0.3:
                return 90
            if price_chg < -0.05 and oi_chg < -0.1:
                return 50
        return 0
    
    def _factor_top_trader(self, sentiment, is_long):
        """Top traders L/S flipped against position."""
        top_ls = sentiment.get('top_ls', 0)
        if not top_ls:
            return 0
        
        if is_long:
            # Top traders now SHORT-biased
            if top_ls <= 0.7:
                return 90
            if top_ls <= 0.9:
                return 50
        else:
            # Top traders now LONG-biased
            if top_ls >= 1.4:
                return 90
            if top_ls >= 1.15:
                return 50
        return 0
    
    def _factor_ls_extreme(self, sentiment, is_long):
        """Crowd L/S ratio at extreme = reversal likely."""
        ls = sentiment.get('ls_ratio', 0)
        if not ls:
            return 0
        
        if is_long:
            # Crowd became very LONG (>2.0) = contrarian SHORT signal
            if ls >= 2.5:
                return 90
            if ls >= 2.0:
                return 60
        else:
            # Crowd became very SHORT (<0.5)
            if ls <= 0.4:
                return 90
            if ls <= 0.5:
                return 60
        return 0
    
    def _factor_wall_approach(self, symbol, price, is_long):
        """Approaching large liquidity wall in profit direction."""
        # Only works for BTC (Liquidity Map is BTC-only)
        if symbol != 'BTCUSDT':
            return 0
        try:
            from detection.liquidity_map import get_liquidity_map
            lm = get_liquidity_map()
            if not lm:
                return 0
            state = lm.get_current_state()
            walls = state.get('walls', [])
            if not walls:
                return 0
            
            # Find nearest wall in profit direction
            for wall in walls:
                wall_price = wall.get('price', 0)
                wall_usd = wall.get('size_usd', 0)
                side = wall.get('side', '')
                
                if not wall_price or wall_usd < 5_000_000:
                    continue
                
                dist_pct = abs(wall_price - price) / price * 100 if price else 100
                
                if is_long and side == 'ask' and wall_price > price and dist_pct < 0.5:
                    # LONG hitting ask wall
                    return 90 if dist_pct < 0.2 else 60
                if not is_long and side == 'bid' and wall_price < price and dist_pct < 0.5:
                    # SHORT hitting bid wall
                    return 90 if dist_pct < 0.2 else 60
        except:
            pass
        return 0
    
    def _factor_rsi_divergence(self, klines, is_long):
        """Simplified RSI divergence on 15-min window."""
        if len(klines) < 30:
            return 0
        
        # Simple RSI approximation on 14 periods
        prices = [c['p'] for c in klines[-30:]]
        gains = []
        losses = []
        for i in range(1, len(prices)):
            chg = prices[i] - prices[i-1]
            gains.append(max(0, chg))
            losses.append(max(0, -chg))
        
        if len(gains) < 14:
            return 0
        
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Compare to earlier RSI
        prev_gain = sum(gains[-28:-14]) / 14 if len(gains) >= 28 else avg_gain
        prev_loss = sum(losses[-28:-14]) / 14 if len(losses) >= 28 else avg_loss
        prev_rsi = 50
        if prev_loss > 0:
            prev_rsi = 100 - (100 / (1 + prev_gain / prev_loss))
        
        if is_long:
            # Bearish divergence: price up, RSI down
            if prices[-1] > prices[-14] and rsi < prev_rsi and rsi > 60:
                return 80 if rsi > 70 else 50
        else:
            # Bullish divergence: price down, RSI up
            if prices[-1] < prices[-14] and rsi > prev_rsi and rsi < 40:
                return 80 if rsi < 30 else 50
        return 0
    
    def _factor_funding_extreme(self, symbol, is_long):
        """Funding rate overheated = squeeze risk."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if not fm:
                return 0
            # Check if symbol is in funding watchlist with extreme values
            rates = fm.get_coin_rates(symbol)
            if not rates.get('found'):
                return 0
            r_list = rates.get('rates', [])
            if not r_list:
                return 0
            current_rate = r_list[-1].get('r', 0)
            
            if is_long:
                # Funding very positive = LONGs paying = overheated
                if current_rate >= 0.05:
                    return 90
                if current_rate >= 0.03:
                    return 50
            else:
                # Funding very negative = SHORTs paying = overheated
                if current_rate <= -0.05:
                    return 90
                if current_rate <= -0.03:
                    return 50
        except:
            pass
        return 0
    
    # ========================================
    # ALERTS
    # ========================================
    
    def _check_alert(self, symbol, pos, analysis):
        if not self._settings.get('telegram_alerts', True) or not self.notifier:
            return
        
        action = analysis.get('action', 'HOLD')
        if action == 'HOLD':
            return
        
        # Only alert on action level change (not every scan)
        last = self._last_alerts.get(symbol, '')
        if last == action:
            return
        
        # Don't alert on downgrade (e.g. FULL_EXIT → WARNING) — only upgrade
        priority = {'HOLD': 0, 'WARNING': 1, 'PARTIAL_EXIT': 2, 'FULL_EXIT': 3}
        if priority.get(action, 0) <= priority.get(last, 0):
            return
        
        self._last_alerts[symbol] = action
        
        try:
            self._send_alert(pos, analysis)
        except Exception as e:
            print(f"[EXIT] Alert error: {e}")
    
    def _send_alert(self, pos, analysis):
        symbol = analysis['symbol']
        side = analysis['side']
        score = analysis['score']
        action = analysis['action']
        pnl = analysis['pnl_pct']
        reasons = analysis.get('reasons', [])
        
        icon = '🚨' if action == 'FULL_EXIT' else '⚠️' if action == 'PARTIAL_EXIT' else '⚠️'
        side_icon = '🟢' if side == 'LONG' else '🔴'
        pnl_str = f"{pnl:+.2f}%"
        pnl_icon = '💚' if pnl > 0 else '❤️'
        
        reasons_str = '\n'.join(f"  • {r}" for r in reasons[:5])
        
        msg = (
            f"{icon} <b>EXIT {action}: {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{side_icon} {side}  {pnl_icon} {pnl_str}\n"
            f"📊 Exit Score: <b>{score}/100</b>\n"
            f"💰 ${analysis.get('current_price', 0):,.6g}\n\n"
            f"<b>Factors:</b>\n{reasons_str}\n\n"
            f"<b>{analysis.get('action_text', '')}</b>\n"
            f"⏱ {analysis['timestamp'][11:16]} UTC"
        )
        
        self.notifier.send_message(msg)
        print(f"[EXIT] 📨 {symbol} {side}: {action} score={score}")
    
    # ========================================
    # HISTORY STORAGE
    # ========================================
    
    def _store_history(self, symbol, analysis):
        if not self.db:
            return
        try:
            date = analysis['timestamp'][:10]
            time_str = analysis['timestamp'][11:16]
            
            key = f"{DB_KEY_HISTORY}_{date}"
            history = self.db.get_setting(key, [])
            if not isinstance(history, list):
                history = []
            
            history.append({
                't': time_str,
                'symbol': symbol,
                'score': analysis.get('score', 0),
                'action': analysis.get('action', 'HOLD'),
                'pnl': analysis.get('pnl_pct', 0),
            })
            
            # Keep last 2000 per day
            if len(history) > 2000:
                history = history[-2000:]
            
            self.db.set_setting(key, history)
        except:
            pass
    
    def get_history(self, date='') -> Dict:
        if not self.db:
            return {'history': [], 'date': date}
        if not date:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        try:
            history = self.db.get_setting(f"{DB_KEY_HISTORY}_{date}", [])
            if not isinstance(history, list):
                history = []
            return {'history': history, 'date': date}
        except:
            return {'history': [], 'date': date}
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_state(self) -> Dict:
        with self._lock:
            # Strip internal fields
            positions = []
            for sym, st in self._state.items():
                clean = {k: v for k, v in st.items() if not k.startswith('_')}
                positions.append(clean)
            
            return {
                'positions': positions,
                'scan_count': self._scan_count,
                'source': self._settings.get('data_source', 'internal'),
                'enabled': self._settings.get('enabled', True),
                'running': self._running,
            }


# Singleton
_instance: Optional[PositionExitMonitor] = None

def get_exit_monitor() -> Optional[PositionExitMonitor]:
    return _instance

def init_exit_monitor(db=None, notifier=None, bybit_connector=None) -> PositionExitMonitor:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = PositionExitMonitor(db=db, notifier=notifier, bybit_connector=bybit_connector)
    return _instance
