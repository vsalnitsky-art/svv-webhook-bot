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
        trend_filter_enabled: bool = False,
        trend_fast_ema: int = 5,
        trend_slow_ema: int = 13,
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
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_fast_ema = trend_fast_ema
        self.trend_slow_ema = trend_slow_ema
        self.on_signal = on_signal
        
        # State
        self._fvg_zones: Dict[str, FVGZone] = {}
        self._lock = threading.Lock()
        self._running = False
        self._watchlist: List[str] = []
        self._price_getter: Optional[Callable] = None
        
        # Trend data per symbol: {'BTCUSDT': {'trend': 'bullish', 'fast': 67500, 'slow': 67200, 'price': 67800}}
        self._symbol_trends: Dict[str, Dict] = {}
        
        # Threads
        self._monitor_thread = None
        self._scanner_thread = None
        self._cleanup_thread = None
        self._monitor_checks = 0
        
        # Stats
        self._stats = {
            'fvg_detected': 0,
            'fvg_retested': 0,
            'fvg_retested_kline': 0,  # retests caught by kline check (not tick polling)
            'fvg_invalidated': 0,
            'fvg_filtered_size': 0,
            'fvg_filtered_overlap': 0,
            'fvg_filtered_trend': 0,
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
        trend_tag = f", trend=EMA({self.trend_fast_ema}/{self.trend_slow_ema})" if self.trend_filter_enabled else ""
        print(f"[FVG] ✅ Started: {len(watchlist)} symbols, TF={self.timeframe}, "
              f"min={self.min_fvg_pct}%, R:R={self.rr_ratio}, "
              f"mitigation={self.mitigation_src}{trend_tag}, active={active} FVGs")
    
    def stop(self):
        self._running = False
        self._save_to_db()
        print("[FVG] ❌ Stopped")
    
    def update_watchlist(self, watchlist: List[str]):
        self._watchlist = watchlist
    
    def reload_settings(self, settings: Dict):
        for key in ('min_fvg_pct', 'rr_ratio', 'sl_buffer_pct', 'timeframe',
                     'max_fvg_per_symbol', 'mitigation_src',
                     'trend_filter_enabled', 'trend_fast_ema', 'trend_slow_ema'):
            if key in settings:
                current = getattr(self, key)
                if isinstance(current, bool):
                    setattr(self, key, bool(settings[key]))
                else:
                    setattr(self, key, type(current)(settings[key]))
        trend_tag = f", trend=EMA({self.trend_fast_ema}/{self.trend_slow_ema})" if self.trend_filter_enabled else ""
        print(f"[FVG] 🔄 Settings: min={self.min_fvg_pct}%, R:R={self.rr_ratio}, "
              f"buffer={self.sl_buffer_pct}%, src={self.mitigation_src}{trend_tag}")
    
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
                    
                    # Already mitigated by subsequent CLOSED candles?
                    # Exclude last candle (forming/unclosed) — len(klines)-1
                    already_mitigated = False
                    for j in range(i + 1, len(klines) - 1):
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
                    
                    # Already mitigated by subsequent CLOSED candles?
                    already_mitigated = False
                    for j in range(i + 1, len(klines) - 1):
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
    # TREND FILTER (EMA-based)
    # ========================================
    
    @staticmethod
    def _compute_ema(closes: List[float], period: int) -> float:
        """Compute EMA (Exponential Moving Average) from close prices."""
        if len(closes) < period:
            return 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period  # SMA seed
        
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _update_trend(self, symbol: str, klines: List[Dict]):
        """Compute Fast/Slow EMA crossover and determine trend direction."""
        if not self.trend_filter_enabled:
            return
        
        min_needed = max(self.trend_fast_ema, self.trend_slow_ema) + 5
        if len(klines) < min_needed:
            return
        
        closes = [k['close'] for k in klines]
        fast_ema = self._compute_ema(closes, self.trend_fast_ema)
        slow_ema = self._compute_ema(closes, self.trend_slow_ema)
        current_price = closes[-1]
        
        if fast_ema <= 0 or slow_ema <= 0:
            return
        
        # Fast EMA > Slow EMA → uptrend, Fast EMA < Slow EMA → downtrend
        trend = 'bullish' if fast_ema > slow_ema else 'bearish'
        
        self._symbol_trends[symbol] = {
            'trend': trend,
            'fast': round(fast_ema, 6),
            'slow': round(slow_ema, 6),
            'price': current_price,
            'spread_pct': round((fast_ema - slow_ema) / slow_ema * 100, 3),
        }
    
    def _check_trend_filter(self, symbol: str, direction: str) -> bool:
        """
        Check if FVG direction aligns with trend (Fast/Slow EMA crossover).
        Returns True if trade is allowed.
        
        Bullish FVG (LONG) → allowed only when Fast EMA > Slow EMA (uptrend)
        Bearish FVG (SHORT) → allowed only when Fast EMA < Slow EMA (downtrend)
        """
        if not self.trend_filter_enabled:
            return True
        
        trend_data = self._symbol_trends.get(symbol)
        if not trend_data:
            return True  # no data yet, allow
        
        return trend_data['trend'] == direction
    
    # ========================================
    # SCANNING
    # ========================================
    
    def _scan_symbol(self, symbol: str) -> int:
        """Scan one symbol for new FVGs + check kline retests."""
        try:
            interval = self.INTERVAL_MAP.get(self.timeframe, '15')
            minutes = self.TF_MINUTES.get(self.timeframe, 15)
            needed = min(1000, int(self.MAX_AGE / 60 / minutes) + 10)
            
            # Ensure enough klines for EMA computation
            if self.trend_filter_enabled:
                needed = max(needed, self.trend_slow_ema + 20)
            
            klines = self.bybit.get_klines(
                symbol=symbol, interval=interval, limit=needed
            )
            
            if not klines or len(klines) < 10:
                return 0
            
            # Update trend data from klines
            self._update_trend(symbol, klines)
            
            # === KLINE-BASED RETEST CHECK ===
            # This catches retests that tick-polling (5s) may miss
            # Check last few CLOSED candles against active FVGs
            self._check_kline_retests(symbol, klines)
            
            new_fvgs = self._detect_fvg_from_klines(symbol, klines)
            
            if new_fvgs:
                with self._lock:
                    # Get only ACTIVE FVGs for this symbol for dedup check
                    # Don't block new zones at same level as traded/invalidated ones
                    existing_for_symbol = [
                        f for f in self._fvg_zones.values()
                        if f.symbol == symbol and f.status in ('waiting', 'entered')
                    ]
                    active_count = len(existing_for_symbol)
                    
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
    
    def _check_kline_retests(self, symbol: str, klines: List[Dict]):
        """
        Kline-based retest detection — catches fast retests missed by tick polling.
        
        TradingView checks retest on CANDLE CLOSE, not on ticks.
        For each active FVG, check if recent CLOSED candles touched the zone
        and closed on the correct side (= successful retest).
        
        Bullish FVG retest: candle wick dipped into zone (low <= zone.high)
                           AND candle closed above zone (close > zone.high)
        Bearish FVG retest: candle wick pierced into zone (high >= zone.low)  
                           AND candle closed below zone (close < zone.low)
        """
        with self._lock:
            active_fvgs = [
                f for f in self._fvg_zones.values()
                if f.symbol == symbol and f.status in ('waiting', 'entered')
            ]
        
        if not active_fvgs or len(klines) < 4:
            return
        
        # Check last N closed candles (exclude forming candle = last one)
        # N = scan_interval / TF_minutes, but at least 2, at most 10
        minutes = self.TF_MINUTES.get(self.timeframe, 15)
        n_candles = max(2, min(10, self.scan_interval // (minutes * 60) + 2))
        # Closed candles: klines[:-1], take last n_candles of those
        closed_candles = klines[-(n_candles + 1):-1]
        
        for fvg in active_fvgs:
            for candle in closed_candles:
                # Skip candles formed BEFORE the FVG was detected
                try:
                    fvg_detected_ts = datetime.fromisoformat(
                        fvg.detected_at.replace('Z', '+00:00')
                    ).timestamp() * 1000
                    if candle['timestamp'] < fvg_detected_ts:
                        continue
                except:
                    pass
                
                c_low = candle['low']
                c_high = candle['high']
                c_close = candle['close']
                
                if fvg.direction == 'bullish':
                    # Bullish FVG: zone is BELOW price
                    # Retest = wick dips into zone, candle closes ABOVE zone
                    wick_touched = c_low <= fvg.high  # wick reached into/through zone
                    close_above = c_close > fvg.high  # closed above zone = bounce
                    
                    # Mitigation: closed below zone bottom
                    if c_close < fvg.low:
                        fvg.status = 'invalidated'
                        self._stats['fvg_invalidated'] += 1
                        break
                    
                    if wick_touched and close_above:
                        print(f"[FVG] 📊 Kline retest detected: {symbol} BULLISH "
                              f"(candle low=${c_low:.4f} into zone ${fvg.low:.4f}-${fvg.high:.4f}, "
                              f"close=${c_close:.4f})")
                        self._stats['fvg_retested_kline'] += 1
                        self._on_retest(fvg, c_close)
                        break
                
                elif fvg.direction == 'bearish':
                    # Bearish FVG: zone is ABOVE price
                    # Retest = wick pierces into zone, candle closes BELOW zone
                    wick_touched = c_high >= fvg.low  # wick reached into zone
                    close_below = c_close < fvg.low   # closed below zone = rejection
                    
                    # Mitigation: closed above zone top
                    if c_close > fvg.high:
                        fvg.status = 'invalidated'
                        self._stats['fvg_invalidated'] += 1
                        break
                    
                    if wick_touched and close_below:
                        print(f"[FVG] 📊 Kline retest detected: {symbol} BEARISH "
                              f"(candle high=${c_high:.4f} into zone ${fvg.low:.4f}-${fvg.high:.4f}, "
                              f"close=${c_close:.4f})")
                        self._stats['fvg_retested_kline'] += 1
                        self._on_retest(fvg, c_close)
                        break
    
    def _scan_all_symbols(self):
        """Scan all watchlist symbols."""
        total_new = 0
        for symbol in self._watchlist:
            total_new += self._scan_symbol(symbol)
            time.sleep(0.5)  # Rate limit protection (Bybit REST)
        
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
        Check FVG retest state machine.
        
        Two detection modes work together:
        1. Tick-based (this method, every 5s) — catches slow retests
        2. Kline-based (_check_kline_retests, every scan) — catches fast retests
        
        State machine: waiting → entered → retested (or invalidated)
        """
        if current_price <= 0:
            return None
        
        zone_size = fvg.high - fvg.low
        # Mitigation buffer: price must go beyond zone by half the FVG size
        mit_buffer = zone_size * 0.5
        
        in_zone = fvg.low <= current_price <= fvg.high
        
        if fvg.status == 'waiting':
            if in_zone:
                return 'entered'
            # Mitigation check WITH buffer
            if fvg.direction == 'bullish' and current_price < (fvg.low - mit_buffer):
                return 'invalidated'
            if fvg.direction == 'bearish' and current_price > (fvg.high + mit_buffer):
                return 'invalidated'
            return None
        
        elif fvg.status == 'entered':
            if in_zone:
                return None
            
            if fvg.direction == 'bullish':
                if current_price > fvg.high:
                    return 'retested'       # Bounced up → LONG
                elif current_price < (fvg.low - mit_buffer):
                    return 'invalidated'    # Mitigated
            
            elif fvg.direction == 'bearish':
                if current_price < fvg.low:
                    return 'retested'       # Rejected down → SHORT
                elif current_price > (fvg.high + mit_buffer):
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
        
        # Guard: don't process already-traded or invalidated FVGs
        if fvg.status not in ('waiting', 'entered'):
            return
        
        # === OPPOSITE FVG FILTER ===
        # If a newer FVG in the opposite direction exists → skip signal
        opposite_direction = 'bearish' if fvg.direction == 'bullish' else 'bullish'
        opposite_fvgs = [
            f for f in self._fvg_zones.values()
            if (f.symbol == fvg.symbol and
                f.direction == opposite_direction and
                f.status in ('waiting', 'entered') and
                f.detected_at > fvg.detected_at)
        ]
        if opposite_fvgs:
            opp = opposite_fvgs[0]
            label = 'LONG' if fvg.direction == 'bullish' else 'SHORT'
            print(f"[FVG] 🚫 OPPOSITE FVG BLOCKED: {fvg.symbol} {label} "
                  f"(newer {opp.direction} FVG at ${opp.low:.4f}-${opp.high:.4f})")
            fvg.status = 'waiting'
            return

        # === TREND FILTER ===
        if not self._check_trend_filter(fvg.symbol, fvg.direction):
            trend_data = self._symbol_trends.get(fvg.symbol, {})
            trend_dir = trend_data.get('trend', '?')
            fast_val = trend_data.get('fast', 0)
            slow_val = trend_data.get('slow', 0)
            label = 'LONG' if fvg.direction == 'bullish' else 'SHORT'
            print(f"[FVG] 🚫 TREND BLOCKED: {fvg.symbol} {label} "
                  f"(FVG={fvg.direction}, trend={trend_dir}, "
                  f"Fast={fast_val:.4f}, Slow={slow_val:.4f})")
            # Keep FVG in 'waiting' so it can be re-entered later if trend changes
            fvg.status = 'waiting'
            self._stats['fvg_filtered_trend'] += 1
            return
        
        # === LIVE PRICE REFRESH ===
        # Retest may have been detected from a closed kline (up to 5 min old).
        # Fetch the latest WS price so entry/SL/TP reflect actual market price
        # at the moment the signal is sent, not the detection candle close.
        kline_price = current_price
        if self._price_getter:
            live_price = self._price_getter(fvg.symbol)
            if live_price > 0:
                current_price = live_price

        sl_price, tp_price = self._calculate_sl_tp(fvg, current_price)

        fvg.status = 'traded'
        fvg.entry_price = current_price
        fvg.sl_price = sl_price
        fvg.tp_price = tp_price

        signal_type = 'BUY' if fvg.direction == 'bullish' else 'SELL'
        label = 'LONG' if signal_type == 'BUY' else 'SHORT'
        risk_pct = abs(current_price - sl_price) / current_price * 100
        price_tag = ''
        if kline_price > 0 and abs(current_price - kline_price) / kline_price > 0.0005:
            price_tag = f' (kline=${kline_price:.4f} → live=${current_price:.4f})'
        else:
            price_tag = ''

        
        trend_tag = ""
        trend_data = self._symbol_trends.get(fvg.symbol)
        if self.trend_filter_enabled and trend_data:
            trend_tag = (f"\n  Trend: {trend_data['trend'].upper()} "
                        f"(Fast={trend_data['fast']:.4f}, Slow={trend_data['slow']:.4f})")
        
        print(f"[FVG] ✅ RETEST SIGNAL: {fvg.symbol} {label}\n"
              f"  FVG: ${fvg.low:.4f} - ${fvg.high:.4f} ({fvg.size_pct}%)\n"
              f"  Entry: ${current_price:.4f}, SL: ${sl_price:.4f}, TP: ${tp_price:.4f}{price_tag}\n"
              f"  Risk: {risk_pct:.2f}%, R:R: {self.rr_ratio}{trend_tag}")
        
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
        
        prices_ok = 0
        prices_zero = 0
        
        for fvg in active_fvgs:
            try:
                current_price = self._price_getter(fvg.symbol)
                if current_price <= 0:
                    prices_zero += 1
                    continue
                
                prices_ok += 1
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
        
        self._monitor_checks += 1
        # Heartbeat every ~5 min (60 checks at 5s interval)
        if self._monitor_checks % 60 == 1:
            entered = sum(1 for f in active_fvgs if f.status == 'entered')
            print(f"[FVG Monitor] ✅ Check #{self._monitor_checks}: "
                  f"{len(active_fvgs)} active ({entered} entered), "
                  f"prices: {prices_ok} ok / {prices_zero} zero")
    
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
            'trend_filter_enabled': self.trend_filter_enabled,
            'trend_fast_ema': self.trend_fast_ema,
            'trend_slow_ema': self.trend_slow_ema,
            'trends': dict(self._symbol_trends),
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
