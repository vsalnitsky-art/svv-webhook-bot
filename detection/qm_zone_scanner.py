"""
QM Zone Scanner v1.0 — Multi-Timeframe Quasimodo Hunter

Архітектура:
    HTF (15m) → SMC Structure Filter → Зони (Strong Low, HL, Weak High, LH)
                                            ↓
                                  Ціна входить в зону?
                                            ↓ YES  
                          LTF (5m) → Quasimodo Detector → Паттерн знайдений?
                                            ↓ YES
                                    📨 Telegram сигнал

Не залежить від talib — використовує чистий numpy.
"""

import time
import threading
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

# Local imports
try:
    from detection.smc_structure_filter import SMCSignalFilter, create_smc_filter, TrendBias
    from detection.qm_detector import QMDetector, QMPattern
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("[QM Scanner] ⚠️ SMC or QM modules not available")


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Kline:
    """Свічка"""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @classmethod
    def from_binance(cls, data: list) -> 'Kline':
        return cls(
            open_time=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            close_time=int(data[6])
        )


@dataclass
class ZoneInfo:
    """Інформація про активну SMC зону"""
    zone_type: str          # 'demand' або 'supply'
    level_name: str         # 'Strong Low', 'HL', 'Weak High', 'LH'
    level_price: float
    distance_pct: float     # Відстань ціни від рівня у %
    trend_bias: str         # 'BULLISH', 'BEARISH', 'NEUTRAL'
    expected_direction: str  # 'BUY' або 'SELL'


@dataclass
class QMSignal:
    """Повний QM сигнал для відправки"""
    symbol: str
    direction: str          # 'BUY' або 'SELL'
    
    # Зона HTF
    zone: ZoneInfo
    htf_timeframe: str
    
    # Паттерн LTF
    pattern: Dict
    ltf_timeframe: str
    
    # Торгові рівні
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    
    # Метрики
    confidence: float
    strength: float
    risk_reward: float
    risk_percent: float
    
    # Ціна
    current_price: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'zone_type': self.zone.zone_type,
            'zone_level': self.zone.level_name,
            'zone_price': self.zone.level_price,
            'zone_distance_pct': round(self.zone.distance_pct, 3),
            'htf_timeframe': self.htf_timeframe,
            'ltf_timeframe': self.ltf_timeframe,
            'trend_bias': self.zone.trend_bias,
            'pattern': self.pattern,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'tp1': self.take_profit_1,
            'tp2': self.take_profit_2,
            'confidence': round(self.confidence, 1),
            'strength': round(self.strength, 1),
            'risk_reward': round(self.risk_reward, 2),
            'risk_percent': round(self.risk_percent, 2),
            'price': self.current_price,
            'timestamp': self.timestamp,
        }

    def format_telegram(self) -> str:
        """Форматувати повідомлення для Telegram"""
        emoji = "🟢" if self.direction == "BUY" else "🔴"
        action = "LONG" if self.direction == "BUY" else "SHORT"
        zone_emoji = "🟢" if self.zone.zone_type == "demand" else "🔴"
        
        msg = f"""{emoji} QM SIGNAL: {action}

📊 {self.symbol}
💰 Ціна: ${self.current_price:,.4f}

🔍 HTF Зона ({self.htf_timeframe}):
{zone_emoji} {self.zone.level_name} @ ${self.zone.level_price:,.4f}
📐 Відстань: {self.zone.distance_pct:.2f}%
📈 Тренд: {self.zone.trend_bias}

🎯 QM Паттерн ({self.ltf_timeframe}):
   A: ${self.pattern['points']['A']:,.4f}
   B: ${self.pattern['points']['B']:,.4f}
   C: ${self.pattern['points']['C']:,.4f} (Head)
   D: ${self.pattern['points']['D']:,.4f}
   E: ${self.pattern['points']['E']:,.4f}

📋 Торгові рівні:
   Entry: ${self.entry:,.4f}
   SL: ${self.stop_loss:,.4f}
   TP1: ${self.take_profit_1:,.4f}
   TP2: ${self.take_profit_2:,.4f}

📊 Метрики:
   Confidence: {self.confidence:.0f}%
   Strength: {self.strength:.0f}%
   R:R = {self.risk_reward:.2f}
   Risk: {self.risk_percent:.2f}%

⏰ {self.timestamp[:19]} UTC"""
        
        return msg


# ============================================
# QM ZONE SCANNER
# ============================================

class QMZoneScanner:
    """
    Multi-Timeframe Quasimodo Zone Scanner
    
    Потік роботи:
    1. Завантажує HTF свічки → будує SMC структуру → визначає зони
    2. Моніторить ціну — чи вона в зоні?
    3. Якщо в зоні → завантажує LTF свічки → шукає QM паттерн
    4. Якщо паттерн знайдений → валідація → сигнал
    """
    
    REST_BASE_URL = "https://api.binance.com/api/v3"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    
    TIMEFRAME_MAP = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
        '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
        '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d'
    }
    
    # Скільки свічок завантажувати для кожного TF
    CANDLES_MAP = {
        '1m': 500, '3m': 400, '5m': 300, '15m': 300,
        '30m': 250, '1h': 500, '2h': 300, '4h': 300,
        '6h': 200, '8h': 200, '12h': 150, '1d': 100
    }

    def __init__(
        self,
        # === HTF Settings (SMC Zones) ===
        htf_timeframe: str = '15m',
        smc_swing_length: int = 50,
        smc_zone_threshold: float = 1.0,
        
        # === LTF Settings (QM Pattern) ===
        ltf_timeframe: str = '5m',
        qm_min_swing_bars: int = 5,
        qm_atr_period: int = 14,
        qm_min_swing_atr: float = 1.2,
        qm_min_pattern_bars: int = 25,
        qm_lookback_bars: int = 150,
        qm_min_db_diff_pct: float = 0.5,
        qm_sl_buffer_pct: float = 0.2,
        qm_min_confidence: float = 70.0,
        qm_min_rr_ratio: float = 1.5,
        qm_max_risk_pct: float = 2.0,
        
        # === Scanner Settings ===
        scan_interval: int = 15,
        
        # === Callbacks ===
        on_signal: Callable = None,
    ):
        # HTF
        self.htf_timeframe = htf_timeframe
        self.smc_swing_length = smc_swing_length
        self.smc_zone_threshold = smc_zone_threshold
        
        # LTF
        self.ltf_timeframe = ltf_timeframe
        
        # QM Detector
        self.qm_detector = QMDetector(
            min_swing_bars=qm_min_swing_bars,
            atr_period=qm_atr_period,
            min_swing_atr=qm_min_swing_atr,
            min_pattern_bars=qm_min_pattern_bars,
            lookback_bars=qm_lookback_bars,
            min_db_diff_pct=qm_min_db_diff_pct,
            sl_buffer_pct=qm_sl_buffer_pct,
            min_confidence=qm_min_confidence,
            min_rr_ratio=qm_min_rr_ratio,
            max_risk_pct=qm_max_risk_pct,
        )
        
        # Scanner
        self.scan_interval = scan_interval
        self.on_signal = on_signal
        
        # State
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Per-symbol caches
        self._htf_candles: Dict[str, List[Kline]] = {}   # symbol -> HTF klines
        self._ltf_candles: Dict[str, List[Kline]] = {}   # symbol -> LTF klines
        self._smc_filters: Dict[str, SMCSignalFilter] = {}  # symbol -> SMC filter
        self._active_zones: Dict[str, ZoneInfo] = {}      # symbol -> active zone
        self._watchlist: List[str] = []
        
        # Signal tracking
        self._last_signals: Dict[str, Tuple[str, float]] = {}  # symbol -> (direction, timestamp)
        self._signal_cooldown = 3600  # 1 год між однаковими сигналами
        
        # Stats
        self._stats = {
            'scans': 0,
            'zones_active': 0,
            'patterns_found': 0,
            'signals_sent': 0,
            'last_scan_ms': 0,
        }
        
        print(f"[QM Scanner] Initialized: HTF={htf_timeframe}, LTF={ltf_timeframe}, "
              f"Zone threshold={smc_zone_threshold}%")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def start(self, symbols: List[str]):
        """Запустити сканер"""
        if self._running:
            return
        
        self._watchlist = [s.upper() for s in symbols]
        
        if not self._watchlist:
            print("[QM Scanner] ❌ Empty watchlist")
            return
        
        print(f"[QM Scanner] Starting for {len(self._watchlist)} symbols...")
        
        # Завантажити дані
        loaded = self._preload_all()
        
        if loaded == 0:
            print("[QM Scanner] ❌ Failed to load any symbols")
            return
        
        # Запустити сканування
        self._running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        
        print(f"[QM Scanner] ✅ Started: {loaded}/{len(self._watchlist)} symbols loaded")
    
    def stop(self):
        """Зупинити сканер"""
        self._running = False
        print("[QM Scanner] Stopped")
    
    def scan_now(self) -> List[Dict]:
        """Примусове сканування (для кнопки Scan Now)"""
        return self._scan_all_symbols()
    
    def add_symbol(self, symbol: str) -> bool:
        """Додати символ"""
        symbol = symbol.upper()
        if symbol not in self._watchlist:
            self._watchlist.append(symbol)
            if self._running:
                return self._load_symbol(symbol)
        return True
    
    def remove_symbol(self, symbol: str):
        """Видалити символ"""
        symbol = symbol.upper()
        if symbol in self._watchlist:
            self._watchlist.remove(symbol)
        with self._lock:
            self._htf_candles.pop(symbol, None)
            self._ltf_candles.pop(symbol, None)
            self._smc_filters.pop(symbol, None)
            self._active_zones.pop(symbol, None)
    
    def get_status(self) -> Dict:
        """Статус сканера"""
        return {
            'running': self._running,
            'watchlist': self._watchlist,
            'htf_timeframe': self.htf_timeframe,
            'ltf_timeframe': self.ltf_timeframe,
            'stats': dict(self._stats),
            'active_zones': {
                sym: {
                    'zone_type': z.zone_type,
                    'level_name': z.level_name,
                    'level_price': z.level_price,
                    'distance_pct': round(z.distance_pct, 3),
                    'expected_direction': z.expected_direction,
                }
                for sym, z in self._active_zones.items()
            },
        }
    
    def get_results(self) -> List[Dict]:
        """Результати сканування для UI"""
        results = []
        
        with self._lock:
            for symbol in self._watchlist:
                htf_klines = self._htf_candles.get(symbol, [])
                ltf_klines = self._ltf_candles.get(symbol, [])
                smc = self._smc_filters.get(symbol)
                zone = self._active_zones.get(symbol)
                
                price = float(htf_klines[-1].close) if htf_klines else 0
                
                # SMC info
                smc_info = {}
                if smc:
                    status = smc.get_status()
                    smc_info = {
                        'trend': status['trend_bias'],
                        'strong_low': status.get('strong_low'),
                        'last_hl': status.get('last_hl'),
                        'weak_high': status.get('weak_high'),
                        'last_lh': status.get('last_lh'),
                        'swing_high': status.get('swing_high'),
                        'swing_low': status.get('swing_low'),
                    }
                
                results.append({
                    'symbol': symbol,
                    'price': price,
                    'htf_candles': len(htf_klines),
                    'ltf_candles': len(ltf_klines),
                    'smc': smc_info,
                    'zone': zone.level_name if zone else None,
                    'zone_type': zone.zone_type if zone else None,
                    'zone_distance': round(zone.distance_pct, 3) if zone else None,
                    'hunting': zone is not None,
                })
        
        return sorted(results, key=lambda x: x['symbol'])
    
    def is_running(self) -> bool:
        return self._running
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    def _preload_all(self) -> int:
        """Завантажити дані для всіх символів"""
        loaded = 0
        for symbol in self._watchlist:
            try:
                if self._load_symbol(symbol):
                    loaded += 1
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"[QM Scanner] Error loading {symbol}: {e}")
        return loaded
    
    def _load_symbol(self, symbol: str) -> bool:
        """Завантажити HTF + LTF дані для символу"""
        try:
            # HTF
            htf_klines = self._fetch_klines(symbol, self.htf_timeframe)
            if not htf_klines or len(htf_klines) < 100:
                print(f"[QM Scanner] ❌ Not enough HTF data for {symbol}: {len(htf_klines) if htf_klines else 0}")
                return False
            
            # LTF
            ltf_klines = self._fetch_klines(symbol, self.ltf_timeframe)
            if not ltf_klines or len(ltf_klines) < self.qm_detector.lookback_bars:
                print(f"[QM Scanner] ❌ Not enough LTF data for {symbol}: {len(ltf_klines) if ltf_klines else 0}")
                return False
            
            # SMC filter for this symbol
            smc = create_smc_filter(
                swing_length=self.smc_swing_length,
                zone_threshold_percent=self.smc_zone_threshold,
                enabled=True
            )
            
            # Initialize SMC with HTF data (BAR-BY-BAR — як Pine Script)
            highs = np.array([k.high for k in htf_klines])
            lows = np.array([k.low for k in htf_klines])
            closes = np.array([k.close for k in htf_klines])
            
            start_idx = self.smc_swing_length + 10
            for i in range(start_idx, len(highs)):
                smc.update_structure(highs[:i+1], lows[:i+1], closes[:i+1])
            
            # Log detected levels
            status = smc.get_status()
            levels = []
            for lname in ('strong_low', 'last_hl', 'weak_high', 'last_lh'):
                val = status.get(lname)
                if val:
                    levels.append(f"{lname}={val:.4f}")
            levels_str = ', '.join(levels) if levels else 'no levels yet'
            
            with self._lock:
                self._htf_candles[symbol] = htf_klines
                self._ltf_candles[symbol] = ltf_klines
                self._smc_filters[symbol] = smc
            
            print(f"[QM Scanner] ✅ {symbol}: HTF={len(htf_klines)}, LTF={len(ltf_klines)}, "
                  f"trend={status['trend_bias']}, {levels_str}")
            return True
            
        except Exception as e:
            print(f"[QM Scanner] ❌ Error loading {symbol}: {e}")
            return False
    
    def _fetch_klines(self, symbol: str, timeframe: str) -> Optional[List[Kline]]:
        """Завантажити свічки з Binance REST API"""
        try:
            limit = self.CANDLES_MAP.get(timeframe, 200)
            url = f"{self.REST_BASE_URL}/klines"
            params = {
                'symbol': symbol,
                'interval': self.TIMEFRAME_MAP.get(timeframe, timeframe),
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            return [Kline.from_binance(k) for k in data]
            
        except Exception as e:
            print(f"[QM Scanner] API error for {symbol} {timeframe}: {e}")
            return None
    
    def _refresh_candles(self, symbol: str):
        """Оновити свічки для символу (періодичне оновлення)"""
        try:
            # HTF
            new_htf = self._fetch_klines(symbol, self.htf_timeframe)
            if new_htf and len(new_htf) >= 100:
                # Rebuild SMC structure bar-by-bar
                smc = create_smc_filter(
                    swing_length=self.smc_swing_length,
                    zone_threshold_percent=self.smc_zone_threshold,
                    enabled=True
                )
                highs = np.array([k.high for k in new_htf])
                lows = np.array([k.low for k in new_htf])
                closes = np.array([k.close for k in new_htf])
                
                start_idx = self.smc_swing_length + 10
                for i in range(start_idx, len(highs)):
                    smc.update_structure(highs[:i+1], lows[:i+1], closes[:i+1])
                
                with self._lock:
                    self._htf_candles[symbol] = new_htf
                    self._smc_filters[symbol] = smc
            
            # LTF (лише якщо ми в зоні або близько)
            zone = self._active_zones.get(symbol)
            if zone or self._should_load_ltf(symbol):
                new_ltf = self._fetch_klines(symbol, self.ltf_timeframe)
                if new_ltf and len(new_ltf) >= self.qm_detector.lookback_bars:
                    with self._lock:
                        self._ltf_candles[symbol] = new_ltf
                        
        except Exception as e:
            print(f"[QM Scanner] Refresh error {symbol}: {e}")
    
    def _should_load_ltf(self, symbol: str) -> bool:
        """Чи потрібно завантажити LTF дані (якщо ціна близько до зони)"""
        smc = self._smc_filters.get(symbol)
        htf = self._htf_candles.get(symbol, [])
        
        if not smc or not htf:
            return False
        
        price = htf[-1].close
        status = smc.get_status()
        threshold = price * (self.smc_zone_threshold * 2 / 100)  # 2x threshold для preload
        
        # Перевіряємо чи ціна близько до будь-якого рівня
        for level_name in ['strong_low', 'last_hl', 'weak_high', 'last_lh']:
            level = status.get(level_name)
            if level and abs(price - level) <= threshold:
                return True
        
        return False
    
    # ========================================
    # SCAN LOOP
    # ========================================
    
    def _scan_loop(self):
        """Головний цикл сканування"""
        print(f"[QM Scanner] Scan loop started (interval={self.scan_interval}s)")
        
        refresh_counter = 0
        refresh_every = max(1, 60 // self.scan_interval)  # Оновлювати свічки раз на хвилину
        log_every = max(1, 120 // self.scan_interval)  # Лог кожні ~2 хвилини
        log_counter = 0
        
        while self._running:
            try:
                start = time.time()
                
                # Періодичне оновлення свічок
                refresh_counter += 1
                if refresh_counter >= refresh_every:
                    refresh_counter = 0
                    for symbol in list(self._watchlist):
                        if not self._running:
                            break
                        self._refresh_candles(symbol)
                        time.sleep(0.1)
                
                # Сканування
                self._scan_all_symbols()
                
                elapsed = (time.time() - start) * 1000
                self._stats['last_scan_ms'] = elapsed
                self._stats['scans'] += 1
                
                # Періодичний лог прогресу
                log_counter += 1
                if log_counter >= log_every:
                    log_counter = 0
                    zones = self._active_zones
                    zone_info = ', '.join(f"{s}={z.level_name}" for s, z in zones.items()) if zones else 'none'
                    print(f"[QM Scanner] 📊 Scan #{self._stats['scans']}: "
                          f"{len(self._watchlist)} symbols, "
                          f"{len(zones)} in zones [{zone_info}], "
                          f"patterns={self._stats['patterns_found']}, "
                          f"signals={self._stats['signals_sent']}, "
                          f"{elapsed:.0f}ms")
                
                # Чекаємо до наступного сканування
                sleep_time = max(1, self.scan_interval - (time.time() - start))
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[QM Scanner] Scan loop error: {e}")
                time.sleep(5)
        
        print("[QM Scanner] Scan loop stopped")
    
    def _scan_all_symbols(self) -> List[Dict]:
        """Сканування всіх символів"""
        results = []
        active_zones = 0
        
        for symbol in list(self._watchlist):
            if not self._running and not results:  # Дозволити scan_now() працювати
                pass
            
            result = self._scan_symbol(symbol)
            if result:
                results.append(result)
            
            if symbol in self._active_zones:
                active_zones += 1
        
        self._stats['zones_active'] = active_zones
        
        # Лог першого сканування
        if self._stats['scans'] == 0:
            zones = self._active_zones
            zone_info = ', '.join(f"{s}={z.level_name}" for s, z in zones.items()) if zones else 'none'
            print(f"[QM Scanner] ✅ First scan complete: "
                  f"{len(self._watchlist)} symbols, "
                  f"{active_zones} in zones [{zone_info}]")
        
        return results
    
    def _scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Повний цикл сканування одного символу:
        1. Визначити SMC зони (HTF)
        2. Перевірити чи ціна в зоні
        3. Якщо так — шукати QM (LTF)
        """
        with self._lock:
            smc = self._smc_filters.get(symbol)
            htf_klines = self._htf_candles.get(symbol, [])
            ltf_klines = self._ltf_candles.get(symbol, [])
        
        if not smc or not htf_klines:
            return None
        
        current_price = htf_klines[-1].close
        
        # === КРОК 1: Перевірка SMC зон (HTF) ===
        zone = self._check_zones(smc, current_price)
        
        with self._lock:
            if zone:
                self._active_zones[symbol] = zone
            else:
                self._active_zones.pop(symbol, None)
        
        if not zone:
            return None  # Ціна не в зоні — нічого не робимо
        
        # === КРОК 2: Пошук QM патерна (LTF) ===
        if len(ltf_klines) < self.qm_detector.lookback_bars:
            return None
        
        highs = np.array([k.high for k in ltf_klines])
        lows = np.array([k.low for k in ltf_klines])
        closes = np.array([k.close for k in ltf_klines])
        
        # direction_hint від зони
        direction_hint = zone.expected_direction
        
        pattern = self.qm_detector.detect(highs, lows, closes, direction_hint=direction_hint)
        
        if pattern is None:
            return None
        
        # === КРОК 3: Дедуплікація ===
        now = time.time()
        last = self._last_signals.get(symbol)
        if last:
            last_dir, last_time = last
            if last_dir == pattern.direction and (now - last_time) < self._signal_cooldown:
                return None
        
        # === КРОК 4: Побудувати та відправити сигнал ===
        self._stats['patterns_found'] += 1
        
        signal = QMSignal(
            symbol=symbol,
            direction=pattern.direction,
            zone=zone,
            htf_timeframe=self.htf_timeframe,
            pattern=pattern.to_dict(),
            ltf_timeframe=self.ltf_timeframe,
            entry=pattern.entry,
            stop_loss=pattern.stop_loss,
            take_profit_1=pattern.take_profit_1,
            take_profit_2=pattern.take_profit_2,
            confidence=pattern.confidence,
            strength=pattern.strength,
            risk_reward=pattern.risk_reward,
            risk_percent=pattern.risk_percent,
            current_price=current_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        # Зберігаємо для дедуплікації
        self._last_signals[symbol] = (pattern.direction, now)
        self._stats['signals_sent'] += 1
        
        # Відправляємо callback
        if self.on_signal:
            try:
                self.on_signal(signal)
            except Exception as e:
                print(f"[QM Scanner] Signal callback error: {e}")
        
        print(f"[QM Scanner] 🎯 SIGNAL: {symbol} {pattern.direction} "
              f"@ ${current_price:,.4f} in {zone.level_name} zone "
              f"(conf={pattern.confidence:.0f}%, R:R={pattern.risk_reward:.2f})")
        
        return signal.to_dict()
    
    # ========================================
    # ZONE DETECTION (HTF)
    # ========================================
    
    def _check_zones(self, smc: SMCSignalFilter, price: float) -> Optional[ZoneInfo]:
        """
        Перевірити чи ціна знаходиться в SMC зоні
        
        Demand зони (BUY): Strong Low, HL
        Supply зони (SELL): Weak High, LH
        """
        status = smc.get_status()
        trend = status['trend_bias']
        threshold_pct = self.smc_zone_threshold
        
        # Перевіряємо demand зони (потенційний BUY)
        demand_levels = [
            ('Strong Low', status.get('strong_low')),
            ('HL', status.get('last_hl')),
        ]
        
        for name, level in demand_levels:
            if level is None or level == 0:
                continue
            
            distance_pct = (price - level) / level * 100
            
            # Ціна має бути БІЛЯ або ТРОХИ ВИЩЕ рівня (demand zone)
            if abs(distance_pct) <= threshold_pct:
                return ZoneInfo(
                    zone_type='demand',
                    level_name=name,
                    level_price=level,
                    distance_pct=distance_pct,
                    trend_bias=trend,
                    expected_direction='BUY',
                )
        
        # Перевіряємо supply зони (потенційний SELL)
        supply_levels = [
            ('Weak High', status.get('weak_high')),
            ('LH', status.get('last_lh')),
        ]
        
        for name, level in supply_levels:
            if level is None or level == 0:
                continue
            
            distance_pct = (price - level) / level * 100
            
            # Ціна має бути БІЛЯ або ТРОХИ НИЖЧЕ рівня (supply zone)
            if abs(distance_pct) <= threshold_pct:
                return ZoneInfo(
                    zone_type='supply',
                    level_name=name,
                    level_price=level,
                    distance_pct=distance_pct,
                    trend_bias=trend,
                    expected_direction='SELL',
                )
        
        return None
    
    # ========================================
    # SETTINGS
    # ========================================
    
    def reload_settings(self, settings: Dict):
        """Оновити налаштування (hot reload)"""
        changed_tf = False
        
        if 'htf_timeframe' in settings and settings['htf_timeframe'] != self.htf_timeframe:
            self.htf_timeframe = settings['htf_timeframe']
            changed_tf = True
        
        if 'ltf_timeframe' in settings and settings['ltf_timeframe'] != self.ltf_timeframe:
            self.ltf_timeframe = settings['ltf_timeframe']
            changed_tf = True
        
        if 'smc_zone_threshold' in settings:
            self.smc_zone_threshold = float(settings['smc_zone_threshold'])
        
        if 'smc_swing_length' in settings:
            self.smc_swing_length = int(settings['smc_swing_length'])
        
        if 'scan_interval' in settings:
            self.scan_interval = int(settings['scan_interval'])
        
        # QM параметри
        qm_params = {}
        qm_keys = [
            ('qm_min_swing_bars', 'min_swing_bars', int),
            ('qm_atr_period', 'atr_period', int),
            ('qm_min_swing_atr', 'min_swing_atr', float),
            ('qm_min_pattern_bars', 'min_pattern_bars', int),
            ('qm_lookback_bars', 'lookback_bars', int),
            ('qm_min_db_diff_pct', 'min_db_diff_pct', float),
            ('qm_sl_buffer_pct', 'sl_buffer_pct', float),
            ('qm_min_confidence', 'min_confidence', float),
            ('qm_min_rr_ratio', 'min_rr_ratio', float),
            ('qm_max_risk_pct', 'max_risk_pct', float),
        ]
        
        for setting_key, attr_name, cast_fn in qm_keys:
            if setting_key in settings:
                setattr(self.qm_detector, attr_name, cast_fn(settings[setting_key]))
        
        if changed_tf and self._running:
            # Перезавантажити дані з новими TF
            print("[QM Scanner] Timeframe changed, reloading data...")
            self._preload_all()
        
        print(f"[QM Scanner] Settings reloaded: HTF={self.htf_timeframe}, LTF={self.ltf_timeframe}")
