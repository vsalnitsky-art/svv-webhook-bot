"""
Zero Lag Trend Service v1.0

Standalone trend detection module based on "Zero Lag Trend Signals (MTF)" by AlgoAlpha.
100% Pine Script match.

Algorithm:
  lag = floor((length - 1) / 2)
  zlema = ema(close + (close - close[lag]), length)
  volatility = highest(atr(length), length*3) * mult
  trend = 1 if crossover(close, zlema + volatility)
  trend = -1 if crossunder(close, zlema - volatility)

Used by: FVG Detector, CTR Scanner (or any module needing trend data).

Architecture:
  - Independent service with own update thread
  - Any module calls check_trend(symbol, direction) → (allowed, block_reason)
  - Shared via ctr_job.py (single instance for all modules)
"""

import time
import threading
from typing import Dict, List, Optional, Callable


class ZeroLagTrendService:
    """
    Standalone Zero Lag Trend service.
    
    Computes ZLEMA + ATR volatility bands for multiple symbols on 3 timeframes (15m, 1h, 4h).
    Any module can query trend state without owning the computation.
    """
    
    # Bybit interval codes
    TF_MAP = {'15m': '15', '1h': '60', '4h': '240'}
    
    # Update frequency per TF (in scan cycles, ~60s each)
    TF_UPDATE_FREQ = {'15': 1, '60': 5, '240': 15}
    
    def __init__(
        self,
        bybit_connector,
        enabled: bool = False,
        tf_15m_enabled: bool = True,
        tf_1h_enabled: bool = True,
        tf_4h_enabled: bool = True,
        length: int = 70,
        mult: float = 1.2,
    ):
        self.bybit = bybit_connector
        self.enabled = enabled
        self.tf_15m_enabled = tf_15m_enabled
        self.tf_1h_enabled = tf_1h_enabled
        self.tf_4h_enabled = tf_4h_enabled
        self.length = length
        self.mult = mult
        
        # State: {symbol: {'15': {trend, zlema, ...}, '60': {...}, '240': {...}}}
        self._trends: Dict[str, Dict[str, Dict]] = {}
        self._lock = threading.RLock()
        
        # Update tracking
        self._scan_counter: int = 0
        self._watchlist: List[str] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_interval: int = 60  # seconds between scans
        
        # Klines provider callback (optional: for 15m reuse from FVG/CTR)
        self._klines_providers: Dict[str, Callable] = {}
    
    # ========================================
    # SETTINGS
    # ========================================
    
    def update_settings(self, **kwargs):
        """Update settings dynamically."""
        for key in ('enabled', 'tf_15m_enabled', 'tf_1h_enabled', 'tf_4h_enabled',
                     'length', 'mult'):
            if key in kwargs:
                setattr(self, key, kwargs[key])
    
    def register_klines_provider(self, tf_key: str, provider: Callable):
        """
        Register external klines source to avoid duplicate REST calls.
        provider(symbol) → List[Dict] or None
        """
        self._klines_providers[tf_key] = provider
    
    # ========================================
    # CORE COMPUTATION (100% Pine Script)
    # ========================================
    
    @staticmethod
    def compute(klines: List[Dict], length: int = 70, mult: float = 1.2) -> Optional[Dict]:
        """
        Compute Zero Lag Trend from kline data.
        
        Pine Script exact:
          lag = floor((length - 1) / 2)
          zlema = ema(src + (src - src[lag]), length)
          volatility = highest(atr(length), length*3) * mult
          trend = 1 if crossover(close, zlema+volatility)
          trend = -1 if crossunder(close, zlema-volatility)
        
        Returns dict with trend, zlema, volatility, upper, lower, price.
        """
        n = len(klines)
        min_needed = length * 3 + length + 10
        if n < min_needed:
            return None
        
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        
        lag = (length - 1) // 2
        
        # Step 1: ZLEMA source = close + (close - close[lag])
        zl_src = [0.0] * n
        for i in range(n):
            zl_src[i] = closes[i] + (closes[i] - closes[max(0, i - lag)]) if i >= lag else closes[i]
        
        # Step 2: EMA of zl_src (standard EMA)
        alpha_ema = 2.0 / (length + 1)
        zlema = [0.0] * n
        zlema[length - 1] = sum(zl_src[:length]) / length  # SMA seed
        for i in range(length, n):
            zlema[i] = alpha_ema * zl_src[i] + (1 - alpha_ema) * zlema[i - 1]
        
        # Step 3: True Range → ATR (Pine ta.atr = RMA = Wilder's smoothing)
        tr = [0.0] * n
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        
        alpha_rma = 1.0 / length  # Wilder's = 1/length
        atr = [0.0] * n
        atr[length - 1] = sum(tr[:length]) / length
        for i in range(length, n):
            atr[i] = alpha_rma * tr[i] + (1 - alpha_rma) * atr[i - 1]
        
        # Step 4: volatility = highest(ATR, length*3) * mult
        window = length * 3
        vol = [0.0] * n
        for i in range(window - 1, n):
            vol[i] = max(atr[max(0, i - window + 1):i + 1]) * mult
        
        # Step 5: Trend state machine (bar-by-bar, Pine exact)
        trend = 0
        start = max(window, length + lag)
        for i in range(start, n):
            upper = zlema[i] + vol[i]
            lower = zlema[i] - vol[i]
            if i > start:
                prev_upper = zlema[i - 1] + vol[i - 1]
                prev_lower = zlema[i - 1] - vol[i - 1]
                # crossover(close, zlema+volatility)
                if closes[i - 1] <= prev_upper and closes[i] > upper:
                    trend = 1
                # crossunder(close, zlema-volatility)
                if closes[i - 1] >= prev_lower and closes[i] < lower:
                    trend = -1
        
        trend_name = 'bullish' if trend == 1 else ('bearish' if trend == -1 else 'neutral')
        return {
            'trend': trend_name,
            'trend_raw': trend,
            'zlema': round(zlema[-1], 6),
            'volatility': round(vol[-1], 6),
            'upper': round(zlema[-1] + vol[-1], 6),
            'lower': round(zlema[-1] - vol[-1], 6),
            'price': closes[-1],
        }
    
    # ========================================
    # UPDATE LOGIC
    # ========================================
    
    def _fetch_klines(self, symbol: str, interval: str) -> Optional[List[Dict]]:
        """Fetch klines via provider (if registered) or Bybit REST."""
        # Try provider first (avoids duplicate REST call)
        provider = self._klines_providers.get(interval)
        if provider:
            try:
                klines = provider(symbol)
                if klines and len(klines) > 250:
                    return klines
            except Exception:
                pass
        
        # Fallback: direct REST
        if not self.bybit:
            return None
        try:
            klines = self.bybit.get_klines(symbol=symbol, interval=interval, limit=1000)
            if klines and len(klines) > 250:
                return klines
        except Exception:
            pass
        return None
    
    def update_symbol(self, symbol: str, klines_15m: Optional[List[Dict]] = None):
        """Update ZLT for one symbol across all enabled TFs."""
        if not self.enabled:
            return
        
        sym_data = self._trends.get(symbol, {})
        
        # 15m — use provided klines or fetch
        if self.tf_15m_enabled:
            if self._scan_counter % self.TF_UPDATE_FREQ['15'] == 0:
                kl = klines_15m
                if not kl or len(kl) < 250:
                    kl = self._fetch_klines(symbol, '15')
                if kl:
                    result = self.compute(kl, self.length, self.mult)
                    if result:
                        sym_data['15'] = result
        
        # 1h
        if self.tf_1h_enabled:
            if self._scan_counter % self.TF_UPDATE_FREQ['60'] == 0:
                kl = self._fetch_klines(symbol, '60')
                if kl:
                    result = self.compute(kl, self.length, self.mult)
                    if result:
                        sym_data['60'] = result
                    time.sleep(0.2)
        
        # 4h
        if self.tf_4h_enabled:
            if self._scan_counter % self.TF_UPDATE_FREQ['240'] == 0:
                kl = self._fetch_klines(symbol, '240')
                if kl:
                    result = self.compute(kl, self.length, self.mult)
                    if result:
                        sym_data['240'] = result
                    time.sleep(0.2)
        
        with self._lock:
            self._trends[symbol] = sym_data
    
    def update_all(self, klines_map_15m: Optional[Dict[str, List[Dict]]] = None):
        """Update all watchlist symbols. Increments scan counter."""
        if not self.enabled or not self._watchlist:
            return
        
        self._scan_counter += 1
        
        for symbol in self._watchlist:
            kl_15m = (klines_map_15m or {}).get(symbol)
            self.update_symbol(symbol, klines_15m=kl_15m)
            time.sleep(0.3)  # Rate limit between symbols
    
    # ========================================
    # TREND CHECK (used by any module)
    # ========================================
    
    def check_trend(self, symbol: str, direction: str) -> tuple:
        """
        Check if direction aligns with ZLT on all enabled TFs.
        
        Returns: (allowed: bool, block_reason: str)
        - allowed=True: all enabled TFs agree (or no data/neutral)
        - allowed=False: at least one TF disagrees
        - block_reason: 'zl_15m', 'zl_1h', or 'zl_4h'
        """
        if not self.enabled:
            return True, ''
        
        with self._lock:
            sym_data = self._trends.get(symbol, {})
        
        checks = [
            (self.tf_15m_enabled, '15', '15m'),
            (self.tf_1h_enabled, '60', '1h'),
            (self.tf_4h_enabled, '240', '4h'),
        ]
        
        for enabled, key, label in checks:
            if not enabled:
                continue
            tf_data = sym_data.get(key)
            if not tf_data or tf_data['trend'] == 'neutral':
                continue  # No data or undetermined → don't block
            if tf_data['trend'] != direction:
                return False, f'zl_{label}'
        
        return True, ''
    
    def get_trend(self, symbol: str, tf_key: str) -> Optional[Dict]:
        """Get trend data for specific symbol and TF key ('15', '60', '240')."""
        with self._lock:
            return self._trends.get(symbol, {}).get(tf_key)
    
    # ========================================
    # STANDALONE UPDATE THREAD
    # ========================================
    
    def start(self, watchlist: List[str], update_interval: int = 60):
        """Start standalone update thread."""
        if self._running:
            return
        
        self._watchlist = list(watchlist)
        self._update_interval = update_interval
        self._running = True
        
        self._thread = threading.Thread(
            target=self._update_loop, daemon=True, name="ZLT-Service"
        )
        self._thread.start()
        
        enabled_tfs = []
        if self.tf_15m_enabled: enabled_tfs.append('15m')
        if self.tf_1h_enabled: enabled_tfs.append('1h')
        if self.tf_4h_enabled: enabled_tfs.append('4h')
        
        print(f"[ZLT] ✅ Started: {len(watchlist)} symbols, "
              f"TFs={','.join(enabled_tfs)}, "
              f"L={self.length}, M={self.mult}")
    
    def stop(self):
        """Stop update thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("[ZLT] ❌ Stopped")
    
    def _update_loop(self):
        """Background loop: update trends periodically."""
        # Initial scan immediately
        self.update_all()
        
        while self._running:
            time.sleep(self._update_interval)
            if not self._running:
                break
            try:
                self.update_all()
            except Exception as e:
                print(f"[ZLT] Update error: {e}")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def get_stats(self) -> Dict:
        """Return full state for UI/API."""
        with self._lock:
            return {
                'enabled': self.enabled,
                'tf_15m_enabled': self.tf_15m_enabled,
                'tf_1h_enabled': self.tf_1h_enabled,
                'tf_4h_enabled': self.tf_4h_enabled,
                'length': self.length,
                'mult': self.mult,
                'trends': dict(self._trends),
                'scan_counter': self._scan_counter,
                'running': self._running,
                'watchlist_size': len(self._watchlist),
            }
    
    def set_watchlist(self, watchlist: List[str]):
        """Update watchlist dynamically."""
        self._watchlist = list(watchlist)
