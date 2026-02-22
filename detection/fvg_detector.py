"""
FVG Detector v1.1 — Fair Value Gap Detection & Retest Trading

Based on TradingView "Объёмные FVG" indicator by vsalnitsky.
Exact Pine Script logic adapted for Python.

Detection rules (Pine Script exact):
  Bullish FVG:
    high[2] < low         — gap exists
    high[2] < high[1]     — middle candle is above prev
    low[2] < low          — middle candle is above prev
    filterFVG             — adaptive size filter (>10% of max FVG in 1000 bars)
    
  Bearish FVG:
    low[2] > high         — gap exists  
    low[2] > low[1]       — middle candle is below prev
    high[2] > high        — middle candle is below prev
    filterFVG             — adaptive size filter

  Bullish zone: top = low (current), bottom = high[2] (2 bars ago)
  Bearish zone: top = low[2] (2 bars ago), bottom = high (current)

Mitigation (invalidation):
  - Bullish FVG: price close < zone.bottom → invalidated
  - Bearish FVG: price close > zone.top → invalidated

State machine for retest trading:
  WAITING     → price enters FVG zone → ENTERED
  ENTERED     → price exits in trend direction → RETESTED (trade!)
  ENTERED     → price punches through (mitigation) → INVALIDATED
  WAITING     → price punches through → INVALIDATED
"""

import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class FVGZone:
    """Один Fair Value Gap"""
    id: str
    symbol: str
    direction: str             # 'bullish' / 'bearish'
    high: float                # top of zone
    low: float                 # bottom of zone
    size_pct: float            # розмір у % від ціни
    candle_time: str           # час середньої свічки (ISO)
    detected_at: str           # коли виявлено (ISO)
    status: str = 'waiting'    # waiting/entered/retested/invalidated/traded/expired
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'FVGZone':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def age_seconds(self) -> float:
        try:
            dt = datetime.fromisoformat(self.detected_at.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - dt).total_seconds()
        except:
            return 0


class FVGDetector:
    """
    FVG Detector — Pine Script accurate detection with retest monitoring.
    
    Uses Bybit Linear Futures klines + WS prices from CTR Scanner.
    """
    
    INTERVAL_MAP = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15',
        '30m': '30', '1h': '60', '4h': '240', '1d': 'D'
    }
    TF_MINUTES = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }
    
    # Lifecycle constants
    MAX_AGE = 48 * 3600         # 48h
    CLEANUP_INTERVAL = 1800     # 30 min
    INVALIDATED_TTL = 3600      # 1h
    TRADED_TTL = 86400          # 24h
    
    def __init__(
        self,
        db,
        bybit_connector,
        timeframe: str = '15m',
        min_fvg_pct: float = 0.1,
        max_fvg_per_symbol: int = 5,
        rr_ratio: float = 1.5,
        sl_buffer_pct: float = 0.2,
        check_interval: int = 5,
        scan_interval: int = 300,
        mitigation_src: str = 'close',  # 'close' or 'highlow'
        on_signal: Optional[Callable] = None,
    ):
        self.db = db
        self.bybit = bybit_connector
        self.timeframe = timeframe
        self.min_fvg_pct = min_fvg_pct
        self.max_fvg_per_symbol = max_fvg_per_symbol
        self.rr_ratio = rr_ratio
        self.sl_buffer_pct = sl_buffer_pct
        self.check_interval = check_interval
        self.scan_interval = scan_interval
        self.mitigation_src = mitigation_src
        self.on_signal = on_signal
        
        # State
        self._fvg_zones: Dict[str, FVGZone] = {}
        self._lock = threading.Lock()
        self._running = False
        self._watchlist: List[str] = []
        self._price_getter: Optional[Callable] = None
        
        # Threads
        self._monitor_thread = None
        self._scanner_thread = None
        self._cleanup_thread = None
        
        # Stats
        self._stats = {
            'fvg_detected': 0,
            'fvg_retested': 0,
            'fvg_invalidated': 0,
            'fvg_filtered_size': 0,
            'fvg_filtered_overlap': 0,
            'last_scan_time': '',
            'scans': 0,
        }
        
        self._load_from_db()
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self, watchlist: List[str], price_getter: Callable):
        if self._running:
            return
        
        self._watchlist = watchlist
        self._price_getter = price_getter
        self._running = True
        
        # Initial scan
        self._scan_all_symbols()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="FVG-Monitor")
        self._monitor_thread.start()
        
        self._scanner_thread = threading.Thread(
            target=self._scanner_loop, daemon=True, name="FVG-Scanner")
        self._scanner_thread.start()
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="FVG-Cleanup")
        self._cleanup_thread.start()
        
        active = sum(1 for f in self._fvg_zones.values() if f.status in ('waiting', 'entered'))
        print(f"[FVG] ✅ Started: {len(watchlist)} symbols, TF={self.timeframe}, "
              f"min={self.min_fvg_pct}%, R:R={self.rr_ratio}, "
              f"mitigation={self.mitigation_src}, active={active} FVGs")
    
    def stop(self):
        self._running = False
        self._save_to_db()
        print("[FVG] ❌ Stopped")
    
    def update_watchlist(self, watchlist: List[str]):
        self._watchlist = watchlist
    
    def reload_settings(self, settings: Dict):
        for key in ('min_fvg_pct', 'rr_ratio', 'sl_buffer_pct', 'timeframe',
                     'max_fvg_per_symbol', 'mitigation_src'):
            if key in settings:
                setattr(self, key, type(getattr(self, key))(settings[key]))
        print(f"[FVG] 🔄 Settings: min={self.min_fvg_pct}%, R:R={self.rr_ratio}, "
              f"buffer={self.sl_buffer_pct}%, src={self.mitigation_src}")
    
    # ========================================
    # FVG DETECTION (Pine Script exact)
    # ========================================
    
    def _detect_fvg_from_klines(self, symbol: str, klines: List[Dict]) -> List[FVGZone]:
        """
        Pine Script exact FVG detection.
        
        Bullish FVG (4 conditions):
          1. high[2] < low       — gap between candle i-2 and candle i
          2. high[2] < high[1]   — middle candle's high is above prev candle's high  
          3. low[2] < low        — confirms upward movement
          4. filterFVG           — adaptive size filter
          
        Bearish FVG (4 conditions):
          1. low[2] > high       — gap between candle i-2 and candle i
          2. low[2] > low[1]     — middle candle's low is below prev candle's low
          3. high[2] > high      — confirms downward movement
          4. filterFVG           — adaptive size filter
        """
        if len(klines) < 3:
            return []
        
        new_fvgs = []
        existing_ids = set(self._fvg_zones.keys())
        
        # Limit scan to MAX_AGE worth of candles
        minutes = self.TF_MINUTES.get(self.timeframe, 15)
        max_candles = min(len(klines) - 1, int(self.MAX_AGE / 60 / minutes) + 3)
        start_idx = max(2, len(klines) - max_candles)
        
        # ---- Pass 1: Collect ALL FVG diff sizes for adaptive filter ----
        all_diffs = []
        for i in range(start_idx, len(klines) - 1):
            h2 = klines[i - 2]['high']
            l2 = klines[i - 2]['low']
            lo = klines[i]['low']
            hi = klines[i]['high']
            c1 = klines[i - 1]['close']
            o1 = klines[i - 1]['open']
            
            # Pine Script diff formula
            if c1 > o1:  # middle candle is bullish
                diff = (lo - h2) / lo * 100 if lo > 0 else 0
            else:
                diff = (l2 - hi) / hi * 100 if hi > 0 else 0
            
            if abs(diff) > 0.001:
                all_diffs.append(abs(diff))
        
        # Percentile 100 (max) for adaptive filter
        p100 = max(all_diffs) if all_diffs else 0
        
        # ---- Pass 2: Detect FVGs with full conditions ----
        for i in range(start_idx, len(klines) - 1):
            c_prev2 = klines[i - 2]  # Pine: [2] bars ago
            c_mid = klines[i - 1]    # Pine: [1] bar ago (middle candle)
            c_curr = klines[i]       # Pine: current bar
            
            ts = str(c_mid['timestamp'])
            
            # Pine Script diff for filter
            if c_mid['close'] > c_mid['open']:
                diff = (c_curr['low'] - c_prev2['high']) / c_curr['low'] * 100 if c_curr['low'] > 0 else 0
            else:
                diff = (c_prev2['low'] - c_curr['high']) / c_curr['high'] * 100 if c_curr['high'] > 0 else 0
            
            # Adaptive filter: sizeFVG = diff / p100 * 100 > 10
            if p100 > 0:
                size_fvg = abs(diff) / p100 * 100
                filter_fvg = size_fvg > 10
            else:
                filter_fvg = abs(diff) >= self.min_fvg_pct
            
            # === Bullish FVG (Pine Script 4 conditions) ===
            is_bull = (
                c_prev2['high'] < c_curr['low'] and      # 1. gap exists
                c_prev2['high'] < c_mid['high'] and       # 2. middle candle high above
                c_prev2['low'] < c_curr['low'] and        # 3. confirms upward
                filter_fvg                                 # 4. size filter
            )
            
            if is_bull:
                fvg_id = f"{symbol}_bull_{ts}"
                if fvg_id not in existing_ids:
                    fvg_top = c_curr['low']
                    fvg_bot = c_prev2['high']
                    mid = (fvg_top + fvg_bot) / 2
                    pct = (fvg_top - fvg_bot) / mid * 100 if mid > 0 else 0
                    
                    # Already mitigated by subsequent candles?
                    already_mitigated = False
                    for j in range(i + 1, len(klines)):
                        val = klines[j]['close'] if self.mitigation_src == 'close' else klines[j]['low']
                        if val < fvg_bot:
                            already_mitigated = True
                            break
                    
                    if not already_mitigated:
                        candle_dt = datetime.fromtimestamp(
                            c_mid['timestamp'] / 1000, tz=timezone.utc
                        ).isoformat()
                        new_fvgs.append(FVGZone(
                            id=fvg_id, symbol=symbol, direction='bullish',
                            high=fvg_top, low=fvg_bot,
                            size_pct=round(pct, 3),
                            candle_time=candle_dt,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                        ))
                    else:
                        self._stats['fvg_filtered_size'] += 1
            
            # === Bearish FVG (Pine Script 4 conditions) ===
            is_bear = (
                c_prev2['low'] > c_curr['high'] and       # 1. gap exists
                c_prev2['low'] > c_mid['low'] and          # 2. middle candle low below
                c_prev2['high'] > c_curr['high'] and       # 3. confirms downward
                filter_fvg                                  # 4. size filter
            )
            
            if is_bear:
                fvg_id = f"{symbol}_bear_{ts}"
                if fvg_id not in existing_ids:
                    fvg_top = c_prev2['low']
                    fvg_bot = c_curr['high']
                    mid = (fvg_top + fvg_bot) / 2
                    pct = (fvg_top - fvg_bot) / mid * 100 if mid > 0 else 0
                    
                    # Already mitigated?
                    already_mitigated = False
                    for j in range(i + 1, len(klines)):
                        val = klines[j]['close'] if self.mitigation_src == 'close' else klines[j]['high']
                        if val > fvg_top:
                            already_mitigated = True
                            break
                    
                    if not already_mitigated:
                        candle_dt = datetime.fromtimestamp(
                            c_mid['timestamp'] / 1000, tz=timezone.utc
                        ).isoformat()
                        new_fvgs.append(FVGZone(
                            id=fvg_id, symbol=symbol, direction='bearish',
                            high=fvg_top, low=fvg_bot,
                            size_pct=round(pct, 3),
                            candle_time=candle_dt,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                        ))
                    else:
                        self._stats['fvg_filtered_size'] += 1
        
        # === Overlap removal (Pine Script logic) ===
        new_fvgs = self._remove_overlapping(new_fvgs)
        
        return new_fvgs
    
    def _remove_overlapping(self, fvgs: List[FVGZone]) -> List[FVGZone]:
        """
        Pine Script overlap removal:
        If FVG_j.top is between FVG_i.bottom and FVG_i.top → remove FVG_i
        """
        if len(fvgs) <= 1:
            return fvgs
        
        to_remove = set()
        for i in range(len(fvgs)):
            if i in to_remove:
                continue
            for j in range(len(fvgs)):
                if i == j or j in to_remove:
                    continue
                if fvgs[j].high < fvgs[i].high and fvgs[j].high > fvgs[i].low:
                    to_remove.add(i)
                    self._stats['fvg_filtered_overlap'] += 1
                    break
        
        return [f for idx, f in enumerate(fvgs) if idx not in to_remove]
    
    # ========================================
    # SCANNING
    # ========================================
    
    def _scan_symbol(self, symbol: str) -> int:
        """Scan one symbol for new FVGs."""
        try:
            interval = self.INTERVAL_MAP.get(self.timeframe, '15')
            minutes = self.TF_MINUTES.get(self.timeframe, 15)
            needed = min(1000, int(self.MAX_AGE / 60 / minutes) + 10)
            
            klines = self.bybit.get_klines(
                symbol=symbol, interval=interval, limit=needed
            )
            
            if not klines or len(klines) < 10:
                return 0
            
            new_fvgs = self._detect_fvg_from_klines(symbol, klines)
            
            if new_fvgs:
                with self._lock:
                    # Get all existing FVGs for this symbol (any status)
                    existing_for_symbol = [
                        f for f in self._fvg_zones.values()
                        if f.symbol == symbol
                    ]
                    active_count = sum(
                        1 for f in existing_for_symbol
                        if f.status in ('waiting', 'entered')
                    )
                    
                    added = 0
                    for fvg in new_fvgs:
                        if active_count + added >= self.max_fvg_per_symbol:
                            break
                        
                        # Skip if overlapping zone already exists (any status)
                        # This prevents re-detection of same gap with different timestamp
                        zone_exists = False
                        for existing in existing_for_symbol:
                            if (existing.direction == fvg.direction and
                                abs(existing.high - fvg.high) / fvg.high < 0.001 and
                                abs(existing.low - fvg.low) / fvg.low < 0.001):
                                zone_exists = True
                                break
                        
                        if zone_exists:
                            continue
                        
                        self._fvg_zones[fvg.id] = fvg
                        existing_for_symbol.append(fvg)
                        added += 1
                        self._stats['fvg_detected'] += 1
                    
                    if added:
                        print(f"[FVG] 📐 {symbol}: +{added} new FVGs "
                              f"(total active: {active_count + added})")
                
                return added
            return 0
            
        except Exception as e:
            print(f"[FVG] Error scanning {symbol}: {e}")
            return 0
    
    def _scan_all_symbols(self):
        """Scan all watchlist symbols."""
        total_new = 0
        for symbol in self._watchlist:
            total_new += self._scan_symbol(symbol)
            time.sleep(0.1)
        
        self._stats['scans'] += 1
        self._stats['last_scan_time'] = datetime.now(timezone.utc).strftime('%H:%M:%S')
        
        active = sum(1 for f in self._fvg_zones.values() if f.status in ('waiting', 'entered'))
        if total_new > 0 or self._stats['scans'] % 6 == 1:
            print(f"[FVG] 📊 Scan #{self._stats['scans']}: "
                  f"+{total_new} new, {active} active FVGs")
        
        self._save_to_db()
    
    # ========================================
    # RETEST MONITORING
    # ========================================
    
    def _check_retest(self, fvg: FVGZone, current_price: float) -> Optional[str]:
        """
        Check FVG retest state with Pine Script mitigation rules.
        
        Mitigation (invalidation):
          - mitigation_src='close':  Bull → close < zone.low;  Bear → close > zone.high
          - mitigation_src='highlow': Bull → low < zone.low;   Bear → high > zone.high
          
        Note: In real-time monitoring we only have last price, so 'close' mode
        uses current price as proxy for close.
        """
        if current_price <= 0:
            return None
        
        in_zone = fvg.low <= current_price <= fvg.high
        
        if fvg.status == 'waiting':
            if in_zone:
                return 'entered'
            # Mitigation check
            if fvg.direction == 'bullish' and current_price < fvg.low:
                return 'invalidated'
            if fvg.direction == 'bearish' and current_price > fvg.high:
                return 'invalidated'
            return None
        
        elif fvg.status == 'entered':
            if in_zone:
                return None
            
            if fvg.direction == 'bullish':
                if current_price > fvg.high:
                    return 'retested'       # Bounced up → LONG
                elif current_price < fvg.low:
                    return 'invalidated'    # Mitigated
            
            elif fvg.direction == 'bearish':
                if current_price < fvg.low:
                    return 'retested'       # Rejected down → SHORT
                elif current_price > fvg.high:
                    return 'invalidated'    # Mitigated
            
            return None
        
        return None
    
    def _calculate_sl_tp(self, fvg: FVGZone, entry_price: float) -> tuple:
        """Calculate SL and TP prices."""
        buffer = entry_price * (self.sl_buffer_pct / 100)
        
        if fvg.direction == 'bullish':
            sl = fvg.low - buffer
            risk = entry_price - sl
            tp = entry_price + (risk * self.rr_ratio)
        else:
            sl = fvg.high + buffer
            risk = sl - entry_price
            tp = entry_price - (risk * self.rr_ratio)
        
        return round(sl, 8), round(tp, 8)
    
    def _on_retest(self, fvg: FVGZone, current_price: float):
        """Handle successful FVG retest → generate trade signal."""
        sl_price, tp_price = self._calculate_sl_tp(fvg, current_price)
        
        fvg.status = 'traded'
        fvg.entry_price = current_price
        fvg.sl_price = sl_price
        fvg.tp_price = tp_price
        
        signal_type = 'BUY' if fvg.direction == 'bullish' else 'SELL'
        label = 'LONG' if signal_type == 'BUY' else 'SHORT'
        risk_pct = abs(current_price - sl_price) / current_price * 100
        
        print(f"[FVG] ✅ RETEST SIGNAL: {fvg.symbol} {label}\n"
              f"  FVG: ${fvg.low:.4f} - ${fvg.high:.4f} ({fvg.size_pct}%)\n"
              f"  Entry: ${current_price:.4f}, SL: ${sl_price:.4f}, TP: ${tp_price:.4f}\n"
              f"  Risk: {risk_pct:.2f}%, R:R: {self.rr_ratio}")
        
        self._stats['fvg_retested'] += 1
        
        if self.on_signal:
            self.on_signal({
                'symbol': fvg.symbol,
                'type': signal_type,
                'price': current_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'reason': f'FVG Retest ({fvg.direction})',
                'fvg_id': fvg.id,
                'fvg_high': fvg.high,
                'fvg_low': fvg.low,
                'fvg_size_pct': fvg.size_pct,
                'rr_ratio': self.rr_ratio,
                'is_fvg': True,
            })
        
        self._save_to_db()
    
    def _monitor_all(self):
        """Check all active FVGs against current prices."""
        if not self._price_getter:
            return
        
        with self._lock:
            active_fvgs = [
                f for f in self._fvg_zones.values()
                if f.status in ('waiting', 'entered')
            ]
        
        for fvg in active_fvgs:
            try:
                current_price = self._price_getter(fvg.symbol)
                if current_price <= 0:
                    continue
                
                new_status = self._check_retest(fvg, current_price)
                
                if new_status:
                    if new_status == 'retested':
                        self._on_retest(fvg, current_price)
                    elif new_status == 'invalidated':
                        fvg.status = 'invalidated'
                        self._stats['fvg_invalidated'] += 1
                    elif new_status == 'entered':
                        fvg.status = 'entered'
                        label = '🟢' if fvg.direction == 'bullish' else '🔴'
                        print(f"[FVG] 🎯 Entered: {fvg.symbol} {label} "
                              f"${fvg.low:.4f}-${fvg.high:.4f} "
                              f"(price=${current_price:.4f})")
                    
            except Exception as e:
                print(f"[FVG] Monitor error {fvg.symbol}: {e}")
    
    # ========================================
    # THREADS
    # ========================================
    
    def _monitor_loop(self):
        while self._running:
            try:
                self._monitor_all()
            except Exception as e:
                print(f"[FVG] Monitor loop error: {e}")
            time.sleep(self.check_interval)
    
    def _scanner_loop(self):
        time.sleep(15)
        while self._running:
            try:
                self._scan_all_symbols()
            except Exception as e:
                print(f"[FVG] Scanner loop error: {e}")
            time.sleep(self.scan_interval)
    
    def _cleanup_loop(self):
        while self._running:
            time.sleep(self.CLEANUP_INTERVAL)
            try:
                self._cleanup()
            except Exception as e:
                print(f"[FVG] Cleanup error: {e}")
    
    def _cleanup(self):
        to_remove = []
        with self._lock:
            for fvg_id, fvg in self._fvg_zones.items():
                age = fvg.age_seconds
                if age > self.MAX_AGE:
                    to_remove.append(fvg_id)
                elif fvg.status == 'invalidated' and age > self.INVALIDATED_TTL:
                    to_remove.append(fvg_id)
                elif fvg.status == 'traded' and age > self.TRADED_TTL:
                    to_remove.append(fvg_id)
                elif fvg.status == 'expired' and age > self.INVALIDATED_TTL:
                    to_remove.append(fvg_id)
            
            for fvg_id in to_remove:
                del self._fvg_zones[fvg_id]
        
        if to_remove:
            print(f"[FVG] 🧹 Cleanup: removed {len(to_remove)} old FVGs")
            self._save_to_db()
    
    # ========================================
    # PERSISTENCE
    # ========================================
    
    def _save_to_db(self):
        try:
            with self._lock:
                data = [fvg.to_dict() for fvg in self._fvg_zones.values()]
            self.db.set_setting('fvg_zones', json.dumps(data))
            self.db.set_setting('fvg_stats', json.dumps(self._stats))
        except Exception as e:
            print(f"[FVG] DB save error: {e}")
    
    def _load_from_db(self):
        try:
            data_str = self.db.get_setting('fvg_zones', '[]')
            data = json.loads(data_str)
            for d in data:
                try:
                    fvg = FVGZone.from_dict(d)
                    if fvg.status in ('waiting', 'entered', 'traded'):
                        self._fvg_zones[fvg.id] = fvg
                except:
                    continue
            
            stats_str = self.db.get_setting('fvg_stats', '{}')
            saved_stats = json.loads(stats_str)
            self._stats.update(saved_stats)
            
            if self._fvg_zones:
                print(f"[FVG] 📋 Loaded {len(self._fvg_zones)} FVGs from DB")
        except Exception as e:
            print(f"[FVG] DB load error: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_zones(self) -> List[Dict]:
        """Get FVG zones for UI (active + recently traded only)."""
        with self._lock:
            zones = []
            for fvg in sorted(self._fvg_zones.values(),
                              key=lambda f: f.detected_at, reverse=True):
                if fvg.status not in ('waiting', 'entered', 'traded', 'retested'):
                    continue
                
                d = fvg.to_dict()
                d['age_min'] = round(fvg.age_seconds / 60, 1)
                
                if self._price_getter:
                    try:
                        price = self._price_getter(fvg.symbol)
                        if price > 0:
                            d['current_price'] = price
                            d['distance_pct'] = round(
                                abs(price - fvg.mid_price) / fvg.mid_price * 100, 2
                            )
                    except:
                        pass
                
                zones.append(d)
            return zones
    
    def get_stats(self) -> Dict:
        with self._lock:
            active = sum(1 for f in self._fvg_zones.values()
                        if f.status in ('waiting', 'entered'))
            total = len(self._fvg_zones)
        return {
            **self._stats,
            'active_fvg': active,
            'total_fvg': total,
            'running': self._running,
            'timeframe': self.timeframe,
            'min_fvg_pct': self.min_fvg_pct,
            'rr_ratio': self.rr_ratio,
            'sl_buffer_pct': self.sl_buffer_pct,
            'mitigation_src': self.mitigation_src,
        }
    
    def clear_zones(self) -> int:
        with self._lock:
            count = len(self._fvg_zones)
            self._fvg_zones.clear()
        self._save_to_db()
        print(f"[FVG] 🗑️ Cleared {count} FVG zones")
        return count
    
    def scan_now(self):
        self._scan_all_symbols()
