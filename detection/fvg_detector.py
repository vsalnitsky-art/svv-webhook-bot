"""
FVG Detector v1.0 — Fair Value Gap Detection & Retest Trading

Архітектура:
1. Завантажує 15m klines з Bybit Linear Futures
2. Знаходить нові FVG (Bullish/Bearish)
3. Моніторить ціну в реальному часі (з WS кешу CTR Scanner)
4. При ретесті + валідному виході → генерує торговий сигнал

Стани FVG:
  WAITING     → ціна ще не входила в зону після створення
  ENTERED     → ціна всередині FVG зони
  RETESTED    → ціна вийшла з FVG у напрямку тренду (TRADE!)
  INVALIDATED → ціна пробила FVG наскрізь (скасовано)
  TRADED      → угоду відкрито, FVG відпрацьований
  EXPIRED     → час життя вичерпано
"""

import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict


# ============================================
# FVG DATA STRUCTURES
# ============================================

@dataclass
class FVGZone:
    """Один Fair Value Gap"""
    id: str                    # unique: symbol_dir_timestamp
    symbol: str
    direction: str             # 'bullish' / 'bearish'
    high: float                # верхня межа FVG
    low: float                 # нижня межа FVG
    size_pct: float            # розмір у % від ціни
    candle_time: str           # час свічки що створила FVG (ISO)
    detected_at: str           # коли виявлено (ISO)
    status: str = 'waiting'    # waiting/entered/retested/invalidated/traded/expired
    entry_price: float = 0.0   # ціна входу (після retest)
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


# ============================================
# FVG DETECTOR
# ============================================

class FVGDetector:
    """
    Детектор Fair Value Gaps з моніторингом ретестів.
    
    Використовує Bybit Linear Futures klines для детекції,
    WS ціни з CTR Scanner для моніторингу в реальному часі.
    """
    
    # Bybit interval mapping
    INTERVAL_MAP = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15',
        '30m': '30', '1h': '60', '4h': '240', '1d': 'D'
    }
    
    # Max klines per request (Bybit limit)
    MAX_KLINES = 1000
    
    # Max age for FVG (seconds)
    MAX_AGE = 48 * 3600  # 48 hours
    
    # Cleanup intervals
    CLEANUP_INTERVAL = 1800  # 30 minutes
    INVALIDATED_TTL = 3600   # 1 hour
    TRADED_TTL = 86400       # 24 hours
    
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
        scan_interval: int = 300,  # new FVG scan every 5 min
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
        self.on_signal = on_signal
        
        # State
        self._fvg_zones: Dict[str, FVGZone] = {}  # id -> FVGZone
        self._lock = threading.Lock()
        self._running = False
        self._watchlist: List[str] = []
        self._price_getter: Optional[Callable] = None  # callback to get current price
        
        # Threads
        self._monitor_thread: Optional[threading.Thread] = None
        self._scanner_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Stats
        self._stats = {
            'fvg_detected': 0,
            'fvg_retested': 0,
            'fvg_invalidated': 0,
            'last_scan_time': '',
            'scans': 0,
        }
        
        # Load saved FVGs from DB
        self._load_from_db()
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self, watchlist: List[str], price_getter: Callable):
        """
        Запустити FVG Detector.
        
        price_getter: callback(symbol) -> float, для отримання поточної ціни з WS кешу
        """
        if self._running:
            return
        
        self._watchlist = watchlist
        self._price_getter = price_getter
        self._running = True
        
        # Initial scan
        self._scan_all_symbols()
        
        # Start monitor thread (checks retests every N seconds)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="FVG-Monitor"
        )
        self._monitor_thread.start()
        
        # Start scanner thread (finds new FVGs periodically)
        self._scanner_thread = threading.Thread(
            target=self._scanner_loop, daemon=True, name="FVG-Scanner"
        )
        self._scanner_thread.start()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="FVG-Cleanup"
        )
        self._cleanup_thread.start()
        
        active = sum(1 for f in self._fvg_zones.values() if f.status in ('waiting', 'entered'))
        print(f"[FVG] ✅ Started: {len(watchlist)} symbols, TF={self.timeframe}, "
              f"min={self.min_fvg_pct}%, R:R={self.rr_ratio}, "
              f"active={active} FVGs")
    
    def stop(self):
        self._running = False
        self._save_to_db()
        print("[FVG] ❌ Stopped")
    
    def update_watchlist(self, watchlist: List[str]):
        self._watchlist = watchlist
    
    def reload_settings(self, settings: Dict):
        """Hot reload from UI settings"""
        if 'min_fvg_pct' in settings:
            self.min_fvg_pct = float(settings['min_fvg_pct'])
        if 'max_fvg_per_symbol' in settings:
            self.max_fvg_per_symbol = int(settings['max_fvg_per_symbol'])
        if 'rr_ratio' in settings:
            self.rr_ratio = float(settings['rr_ratio'])
        if 'sl_buffer_pct' in settings:
            self.sl_buffer_pct = float(settings['sl_buffer_pct'])
        if 'timeframe' in settings:
            self.timeframe = settings['timeframe']
        print(f"[FVG] 🔄 Settings reloaded: min={self.min_fvg_pct}%, "
              f"R:R={self.rr_ratio}, buffer={self.sl_buffer_pct}%")
    
    # ========================================
    # FVG DETECTION (from klines)
    # ========================================
    
    def _detect_fvg_from_klines(self, symbol: str, klines: List[Dict]) -> List[FVGZone]:
        """
        Detect FVGs from candle data.
        
        Bullish FVG: candle[i-2].high < candle[i].low (gap up)
        Bearish FVG: candle[i-2].low > candle[i].high (gap down)
        
        Only detects NEW FVGs (not already in self._fvg_zones).
        """
        if len(klines) < 3:
            return []
        
        new_fvgs = []
        existing_ids = set(self._fvg_zones.keys())
        
        # Scan last portion of klines (recent candles)
        # Skip the last candle (it's still forming)
        scan_range = range(2, len(klines) - 1)
        
        for i in scan_range:
            c_prev2 = klines[i - 2]  # 2 candles ago
            c_prev1 = klines[i - 1]  # 1 candle ago (middle - creates the gap)
            c_curr = klines[i]       # current candle
            
            ts = str(c_prev1['timestamp'])
            
            # === Bullish FVG ===
            if c_prev2['high'] < c_curr['low']:
                fvg_id = f"{symbol}_bull_{ts}"
                if fvg_id not in existing_ids:
                    fvg_high = c_curr['low']
                    fvg_low = c_prev2['high']
                    mid = (fvg_high + fvg_low) / 2
                    size_pct = (fvg_high - fvg_low) / mid * 100 if mid > 0 else 0
                    
                    if size_pct >= self.min_fvg_pct:
                        candle_dt = datetime.fromtimestamp(
                            c_prev1['timestamp'] / 1000, tz=timezone.utc
                        ).isoformat()
                        
                        new_fvgs.append(FVGZone(
                            id=fvg_id,
                            symbol=symbol,
                            direction='bullish',
                            high=fvg_high,
                            low=fvg_low,
                            size_pct=round(size_pct, 3),
                            candle_time=candle_dt,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                            status='waiting',
                        ))
            
            # === Bearish FVG ===
            if c_prev2['low'] > c_curr['high']:
                fvg_id = f"{symbol}_bear_{ts}"
                if fvg_id not in existing_ids:
                    fvg_high = c_prev2['low']
                    fvg_low = c_curr['high']
                    mid = (fvg_high + fvg_low) / 2
                    size_pct = (fvg_high - fvg_low) / mid * 100 if mid > 0 else 0
                    
                    if size_pct >= self.min_fvg_pct:
                        candle_dt = datetime.fromtimestamp(
                            c_prev1['timestamp'] / 1000, tz=timezone.utc
                        ).isoformat()
                        
                        new_fvgs.append(FVGZone(
                            id=fvg_id,
                            symbol=symbol,
                            direction='bearish',
                            high=fvg_high,
                            low=fvg_low,
                            size_pct=round(size_pct, 3),
                            candle_time=candle_dt,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                            status='waiting',
                        ))
        
        return new_fvgs
    
    def _scan_symbol(self, symbol: str) -> int:
        """Scan one symbol for new FVGs. Returns count of new FVGs found."""
        try:
            interval = self.INTERVAL_MAP.get(self.timeframe, '15')
            klines = self.bybit.get_klines(
                symbol=symbol,
                interval=interval,
                limit=self.MAX_KLINES
            )
            
            if not klines or len(klines) < 10:
                return 0
            
            new_fvgs = self._detect_fvg_from_klines(symbol, klines)
            
            if new_fvgs:
                with self._lock:
                    # Count existing active FVGs for this symbol
                    active_count = sum(
                        1 for f in self._fvg_zones.values()
                        if f.symbol == symbol and f.status in ('waiting', 'entered')
                    )
                    
                    added = 0
                    for fvg in new_fvgs:
                        if active_count + added >= self.max_fvg_per_symbol:
                            break
                        self._fvg_zones[fvg.id] = fvg
                        added += 1
                        self._stats['fvg_detected'] += 1
                    
                    if added:
                        print(f"[FVG] 📐 {symbol}: +{added} new FVGs "
                              f"(total active: {active_count + added})")
                
                return len(new_fvgs)
            return 0
            
        except Exception as e:
            print(f"[FVG] Error scanning {symbol}: {e}")
            return 0
    
    def _scan_all_symbols(self):
        """Scan all watchlist symbols for new FVGs."""
        total_new = 0
        for symbol in self._watchlist:
            total_new += self._scan_symbol(symbol)
            time.sleep(0.1)  # Rate limit
        
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
        Check FVG retest state.
        
        Returns new status or None if unchanged.
        
        State machine:
        WAITING → price enters FVG zone → ENTERED
        ENTERED → price exits in trend direction → RETESTED (trade signal!)
        ENTERED → price punches through FVG completely → INVALIDATED
        """
        if current_price <= 0:
            return None
        
        in_zone = fvg.low <= current_price <= fvg.high
        
        if fvg.status == 'waiting':
            if in_zone:
                return 'entered'
            # Check if price already blew through (invalidation without entering)
            if fvg.direction == 'bullish' and current_price < fvg.low:
                return 'invalidated'
            if fvg.direction == 'bearish' and current_price > fvg.high:
                return 'invalidated'
            return None
        
        elif fvg.status == 'entered':
            if in_zone:
                return None  # Still inside
            
            # Exited — check direction
            if fvg.direction == 'bullish':
                if current_price > fvg.high:
                    return 'retested'      # Valid: bounced up from bullish FVG
                elif current_price < fvg.low:
                    return 'invalidated'   # Invalid: fell through
            
            elif fvg.direction == 'bearish':
                if current_price < fvg.low:
                    return 'retested'      # Valid: rejected down from bearish FVG
                elif current_price > fvg.high:
                    return 'invalidated'   # Invalid: broke through
            
            return None
        
        return None
    
    def _calculate_sl_tp(self, fvg: FVGZone, entry_price: float) -> tuple:
        """Calculate SL and TP prices."""
        buffer = entry_price * (self.sl_buffer_pct / 100)
        
        if fvg.direction == 'bullish':
            # LONG: SL below FVG low, TP above entry
            sl = fvg.low - buffer
            risk = entry_price - sl
            tp = entry_price + (risk * self.rr_ratio)
        else:
            # SHORT: SL above FVG high, TP below entry
            sl = fvg.high + buffer
            risk = sl - entry_price
            tp = entry_price - (risk * self.rr_ratio)
        
        return round(sl, 8), round(tp, 8)
    
    def _on_retest(self, fvg: FVGZone, current_price: float):
        """Handle successful FVG retest — generate trade signal."""
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
        
        # Callback to CTR Job
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
                    old_status = fvg.status
                    
                    if new_status == 'retested':
                        self._on_retest(fvg, current_price)
                    elif new_status == 'invalidated':
                        fvg.status = 'invalidated'
                        self._stats['fvg_invalidated'] += 1
                        label = '🟢' if fvg.direction == 'bullish' else '🔴'
                        print(f"[FVG] ❌ Invalidated: {fvg.symbol} {label} "
                              f"${fvg.low:.4f}-${fvg.high:.4f}")
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
        """Retest monitoring every N seconds."""
        while self._running:
            try:
                self._monitor_all()
            except Exception as e:
                print(f"[FVG] Monitor loop error: {e}")
            time.sleep(self.check_interval)
    
    def _scanner_loop(self):
        """Periodic new FVG detection."""
        # Wait a bit before first scan (let WS connect)
        time.sleep(15)
        while self._running:
            try:
                self._scan_all_symbols()
            except Exception as e:
                print(f"[FVG] Scanner loop error: {e}")
            time.sleep(self.scan_interval)
    
    def _cleanup_loop(self):
        """Periodic cleanup of old/invalid FVGs."""
        while self._running:
            time.sleep(self.CLEANUP_INTERVAL)
            try:
                self._cleanup()
            except Exception as e:
                print(f"[FVG] Cleanup error: {e}")
    
    def _cleanup(self):
        """Remove expired, old, and invalid FVGs."""
        now = time.time()
        to_remove = []
        
        with self._lock:
            for fvg_id, fvg in self._fvg_zones.items():
                age = fvg.age_seconds
                
                # Remove expired (>48h)
                if age > self.MAX_AGE:
                    to_remove.append(fvg_id)
                    continue
                
                # Remove old invalidated (>1h)
                if fvg.status == 'invalidated' and age > self.INVALIDATED_TTL:
                    to_remove.append(fvg_id)
                    continue
                
                # Remove old traded (>24h)
                if fvg.status == 'traded' and age > self.TRADED_TTL:
                    to_remove.append(fvg_id)
                    continue
                
                # Remove old expired
                if fvg.status == 'expired' and age > self.INVALIDATED_TTL:
                    to_remove.append(fvg_id)
                    continue
            
            for fvg_id in to_remove:
                del self._fvg_zones[fvg_id]
        
        if to_remove:
            print(f"[FVG] 🧹 Cleanup: removed {len(to_remove)} old FVGs")
            self._save_to_db()
    
    # ========================================
    # PERSISTENCE
    # ========================================
    
    def _save_to_db(self):
        """Save all FVGs to DB."""
        try:
            with self._lock:
                data = [fvg.to_dict() for fvg in self._fvg_zones.values()]
            self.db.set_setting('fvg_zones', json.dumps(data))
            self.db.set_setting('fvg_stats', json.dumps(self._stats))
        except Exception as e:
            print(f"[FVG] DB save error: {e}")
    
    def _load_from_db(self):
        """Load FVGs from DB."""
        try:
            data_str = self.db.get_setting('fvg_zones', '[]')
            data = json.loads(data_str)
            
            for d in data:
                try:
                    fvg = FVGZone.from_dict(d)
                    # Only load active ones
                    if fvg.status in ('waiting', 'entered', 'traded'):
                        self._fvg_zones[fvg.id] = fvg
                except:
                    continue
            
            # Load stats
            stats_str = self.db.get_setting('fvg_stats', '{}')
            saved_stats = json.loads(stats_str)
            self._stats.update(saved_stats)
            
            if self._fvg_zones:
                print(f"[FVG] 📋 Loaded {len(self._fvg_zones)} FVGs from DB")
        except Exception as e:
            print(f"[FVG] DB load error: {e}")
    
    # ========================================
    # PUBLIC API (for UI / routes)
    # ========================================
    
    def get_zones(self) -> List[Dict]:
        """Get all FVG zones for UI display."""
        with self._lock:
            zones = []
            for fvg in sorted(self._fvg_zones.values(),
                              key=lambda f: f.detected_at, reverse=True):
                d = fvg.to_dict()
                d['age_min'] = round(fvg.age_seconds / 60, 1)
                
                # Add current price info
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
        """Get detector statistics."""
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
        }
    
    def clear_zones(self) -> int:
        """Clear all FVG zones."""
        with self._lock:
            count = len(self._fvg_zones)
            self._fvg_zones.clear()
        self._save_to_db()
        print(f"[FVG] 🗑️ Cleared {count} FVG zones")
        return count
    
    def scan_now(self):
        """Manual trigger for scanning."""
        self._scan_all_symbols()
