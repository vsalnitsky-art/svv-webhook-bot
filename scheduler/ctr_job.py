"""
CTR Background Job v2.5 - Fast Edition + Auto-Trade

Integration with CTRFastScanner v2.5 + Bybit Futures Trading.

Changes from v2.4:
- Trade Executor integration: auto-trade on Bybit Linear Futures
- Trade symbols management: mark individual watchlist symbols for trading
- Trade settings: leverage, deposit %, TP/SL, max positions
- Trade log: all operations saved in DB
"""

import threading
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from detection.ctr_scanner_fast import CTRFastScanner
from alerts.telegram_notifier import get_notifier
from storage.db_operations import DBOperations

# Trade executor (optional — works without Bybit keys)
try:
    from trading.ctr_trade_executor import CTRTradeExecutor
    TRADE_AVAILABLE = True
except ImportError:
    TRADE_AVAILABLE = False
    print("[CTR Job] ⚠️ Trade executor not available")

# FVG Detector (optional)
try:
    from detection.fvg_detector import FVGDetector
    FVG_AVAILABLE = True
except ImportError:
    FVG_AVAILABLE = False
    print("[CTR Job] ⚠️ FVG Detector not available")

# SMCTrendFilter is now built-in to CTRFastScanner (no separate import needed)


class CTRFastJob:
    """
    Background job для CTR Fast Scanner
    
    Особливості:
    - Використовує WebSocket для real-time даних
    - Сигнали за 1-5 секунд
    - SMC Structure Filter для фільтрації
    - Smart Reversal Detection (Gap Fill + Trend Guard)
    - Автоматичне збереження результатів в БД
    """
    
    def __init__(self, db: DBOperations):
        self.db = db
        self._scanner: Optional[CTRFastScanner] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Last signal direction per symbol for deduplication
        self._last_signal_direction: Dict[str, str] = {}  # symbol -> 'BUY'/'SELL'
        
        # Virtual positions for SL tracking (works without Auto-Trade)
        self._virtual_positions: Dict[str, Dict] = {}  # symbol -> {direction, entry_price, timestamp}
        self._sl_cooldown: Dict[str, float] = {}  # symbol -> timestamp of last SL trigger
        self._sl_monitor_thread: Optional[threading.Thread] = None
        
        # FVG Detector
        self._fvg_detector: Optional['FVGDetector'] = None
        
        # Trade executor (Bybit Futures)
        self._trade_executor: Optional['CTRTradeExecutor'] = None
        if TRADE_AVAILABLE:
            self._init_trade_executor()
        
        # SMC Trend Filter (HTF)
        # Load settings
        self._load_settings()
        
        # Load last signal directions from DB for persistence across restarts
        self._load_last_directions()
        
        # Load virtual positions from saved signals
        self._load_virtual_positions()
    
    def _init_trade_executor(self):
        """Ініціалізувати торговий модуль"""
        try:
            from core.bybit_connector import get_connector
            connector = get_connector()
            self._trade_executor = CTRTradeExecutor(self.db, connector)
            print("[CTR Job] ✅ Trade executor initialized")
        except Exception as e:
            print(f"[CTR Job] ⚠️ Trade executor init failed: {e}")
            self._trade_executor = None
    
    def _load_settings(self):
        """Завантажити налаштування з БД"""
        self.timeframe = self.db.get_setting('ctr_timeframe', '15m')
        self.fast_length = int(self.db.get_setting('ctr_fast_length', '21'))
        self.slow_length = int(self.db.get_setting('ctr_slow_length', '50'))
        self.cycle_length = int(self.db.get_setting('ctr_cycle_length', '10'))
        self.d1_length = int(self.db.get_setting('ctr_d1_length', '3'))
        self.d2_length = int(self.db.get_setting('ctr_d2_length', '3'))
        self.upper = float(self.db.get_setting('ctr_upper', '75'))
        self.lower = float(self.db.get_setting('ctr_lower', '25'))
        
        # SMC Filter settings
        smc_enabled_str = self.db.get_setting('ctr_smc_filter_enabled', '0')
        self.smc_filter_enabled = smc_enabled_str in ('1', 'true', 'True', 'yes')
        self.smc_swing_length = int(self.db.get_setting('ctr_smc_swing_length', '50'))
        self.smc_zone_threshold = float(self.db.get_setting('ctr_smc_zone_threshold', '1.0'))
        smc_require_trend_str = self.db.get_setting('ctr_smc_require_trend', '1')
        self.smc_require_trend = smc_require_trend_str in ('1', 'true', 'True', 'yes')
        
        # Optional signal filters (OFF = 100% Pine Script original)
        tg_str = self.db.get_setting('ctr_use_trend_guard', '0')
        self.use_trend_guard = tg_str in ('1', 'true', 'True', 'yes')
        gd_str = self.db.get_setting('ctr_use_gap_detection', '0')
        self.use_gap_detection = gd_str in ('1', 'true', 'True', 'yes')
        cd_str = self.db.get_setting('ctr_use_cooldown', '1')
        self.use_cooldown = cd_str in ('1', 'true', 'True', 'yes')
        self.cooldown_seconds = int(self.db.get_setting('ctr_cooldown_seconds', '300'))
        
        # SMC Trend Filter (HTF direction — 4h/1h)
        _b = lambda k, d='0': self.db.get_setting(k, d) in ('1', 'true', 'True', 'yes')
        self.smc_trend_enabled = _b('ctr_smc_trend_enabled', '0')
        self.smc_trend_swing_4h = int(self.db.get_setting('ctr_smc_trend_swing_4h', '50'))
        self.smc_trend_swing_1h = int(self.db.get_setting('ctr_smc_trend_swing_1h', '50'))
        self.smc_trend_mode = self.db.get_setting('ctr_smc_trend_mode', 'both')
        self.smc_trend_refresh = int(self.db.get_setting('ctr_smc_trend_refresh', '900'))
        self.smc_trend_block_neutral = _b('ctr_smc_trend_block_neutral', '0')
        self.smc_trend_early_warning = _b('ctr_smc_trend_early_warning', '0')
        self.smc_trend_swing_15m = int(self.db.get_setting('ctr_smc_trend_swing_15m', '20'))
        
        # Telegram notification mode: 'all' or 'trade_only'
        self.telegram_mode = self.db.get_setting('ctr_telegram_mode', 'all')
        
        # SL Monitor
        self.sl_monitor_enabled = _b('ctr_sl_monitor_enabled', '0')
        self.sl_monitor_pct = float(self.db.get_setting('ctr_sl_monitor_pct', '0'))
        self.sl_check_interval = int(self.db.get_setting('ctr_sl_check_interval', '5'))
        
        # FVG Detector
        self.fvg_enabled = _b('ctr_fvg_enabled', '0')
        self.fvg_timeframe = self.db.get_setting('ctr_fvg_timeframe', '15m')
        self.fvg_min_pct = float(self.db.get_setting('ctr_fvg_min_pct', '0.1'))
        self.fvg_max_per_symbol = int(self.db.get_setting('ctr_fvg_max_per_symbol', '5'))
        self.fvg_rr_ratio = float(self.db.get_setting('ctr_fvg_rr_ratio', '1.5'))
        self.fvg_sl_buffer_pct = float(self.db.get_setting('ctr_fvg_sl_buffer_pct', '0.2'))
        self.fvg_scan_interval = int(self.db.get_setting('ctr_fvg_scan_interval', '300'))
        
        # Watchlist
        watchlist_str = self.db.get_setting('ctr_watchlist', '')
        self.watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
    
    def _load_last_directions(self):
        """Завантажити останні напрямки сигналів з БД для дедуплікації"""
        try:
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            
            for sig in signals:
                symbol = sig.get('symbol')
                sig_type = sig.get('type')
                if symbol and sig_type:
                    self._last_signal_direction[symbol] = sig_type
            
            if self._last_signal_direction:
                print(f"[CTR Job] 📋 Loaded last directions for {len(self._last_signal_direction)} symbols")
                for sym, direction in self._last_signal_direction.items():
                    print(f"  {sym}: {direction}")
        except Exception as e:
            print(f"[CTR Job] Error loading last directions: {e}")
    
    def _load_virtual_positions(self):
        """Reconstruct virtual positions from saved signals for SL tracking"""
        try:
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            
            # Last signal per symbol = current virtual position
            for sig in signals:
                symbol = sig.get('symbol')
                price = sig.get('price', 0)
                sig_type = sig.get('type')
                if symbol and sig_type and price > 0:
                    self._virtual_positions[symbol] = {
                        'direction': sig_type,
                        'entry_price': price,
                        'timestamp': sig.get('timestamp', '')
                    }
            
            if self._virtual_positions:
                print(f"[CTR Job] 📊 Loaded {len(self._virtual_positions)} virtual positions for SL monitor")
        except Exception as e:
            print(f"[CTR Job] Error loading virtual positions: {e}")
    
    def _is_duplicate_signal(self, symbol: str, signal_type: str) -> bool:
        """
        Перевіряє чи сигнал дублікат останнього.
        
        BUY → BUY = ДУБЛІКАТ (ігноруємо)
        BUY → SELL = НОВИЙ (відправляємо)
        SELL → SELL = ДУБЛІКАТ (ігноруємо)
        SELL → BUY = НОВИЙ (відправляємо)
        None → будь-який = НОВИЙ (перший сигнал)
        """
        last_direction = self._last_signal_direction.get(symbol)
        
        if last_direction is None:
            return False  # Перший сигнал для цієї монети
        
        return last_direction == signal_type
    
    def _on_signal(self, signal: Dict):
        """Callback при отриманні сигналу від сканера"""
        try:
            symbol = signal['symbol']
            signal_type = signal['type']
            reason = signal.get('reason', '')
            
            # v2.4: Trend Guard signals are priority — they bypass dedup
            # SL signals and FVG signals are also priority
            is_priority = ("Trend Guard" in reason or signal.get('is_sl', False)
                          or signal.get('is_fvg', False))
            
            # === ДЕДУПЛІКАЦІЯ ===
            if not is_priority and self._is_duplicate_signal(symbol, signal_type):
                print(f"[CTR Job] ⏭️ Duplicate signal skipped: {symbol} {signal_type} "
                      f"(last was also {signal_type}, waiting for opposite)")
                return
            
            # Оновлюємо останній напрямок
            self._last_signal_direction[symbol] = signal_type
            
            # Відправка в Telegram (з урахуванням telegram_mode)
            notifier = get_notifier()
            send_telegram = False
            if notifier:
                if self.telegram_mode == 'trade_only':
                    # Тільки символи з увімкненим Trade
                    trade_symbols = []
                    if self._trade_executor:
                        trade_symbols = self._trade_executor.get_trade_symbols()
                    if symbol in trade_symbols:
                        send_telegram = True
                    else:
                        print(f"[CTR Job] 📵 Telegram skipped: {symbol} {signal_type} (not in trade list)")
                else:
                    send_telegram = True
            
            if send_telegram and notifier:
                msg = signal.get('message', '')
                if not msg and signal.get('is_fvg'):
                    # Build message for FVG signal
                    direction = '🟢 LONG' if signal_type == 'BUY' else '🔴 SHORT'
                    msg = (f"{'=' * 40}\n"
                           f"📐 FVG Retest Signal\n"
                           f"Монета: {symbol}\n"
                           f"{direction} @ ${signal.get('price', 0):.4f}\n"
                           f"FVG: ${signal.get('fvg_low', 0):.4f} - ${signal.get('fvg_high', 0):.4f}\n"
                           f"SL: ${signal.get('sl_price', 0):.4f}\n"
                           f"TP: ${signal.get('tp_price', 0):.4f}\n"
                           f"R:R: {signal.get('rr_ratio', 1.5)}\n"
                           f"{'=' * 40}")
                if msg:
                    notifier.send_message(msg)
            
            # Збереження в БД
            self._save_signal(signal)
            
            # Оновити віртуальну позицію для SL моніторингу
            if signal.get('price', 0) > 0:
                self._virtual_positions[symbol] = {
                    'direction': signal_type,
                    'entry_price': signal['price'],
                    'timestamp': time.time()
                }
            
            smc_tag = " [SMC✓]" if signal.get('smc_filtered') else ""
            reason_tag = f" [{reason}]" if reason else ""
            print(f"[CTR Job] 📨 Signal sent: {symbol} {signal_type}{smc_tag}{reason_tag}")
            
            # === AUTO-TRADE ===
            if self._trade_executor:
                try:
                    # When our SL monitor is active, tell executor to skip Bybit's native SL
                    skip_native_sl = self.sl_monitor_enabled and self.sl_monitor_pct > 0
                    
                    # FVG signals provide their own SL/TP
                    override_sl = signal.get('sl_price', 0) if signal.get('is_fvg') else 0
                    override_tp = signal.get('tp_price', 0) if signal.get('is_fvg') else 0
                    
                    # FVG always sets its own SL on exchange
                    if signal.get('is_fvg') and override_sl > 0:
                        skip_native_sl = False
                    
                    trade_result = self._trade_executor.execute_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=signal.get('price', 0),
                        reason=reason,
                        skip_native_sl=skip_native_sl,
                        override_sl=override_sl,
                        override_tp=override_tp,
                    )
                    
                    if trade_result['success']:
                        trade_msg = f"[CTR Job] 💰 Trade executed: {symbol} {trade_result['action']} — {trade_result['details']}"
                        print(trade_msg)
                        
                        # Notify via Telegram about trade
                        if notifier:
                            emoji = "💰" if trade_result['action'] == 'opened' else "🔄"
                            notifier.send_message(
                                f"{emoji} AUTO-TRADE: {symbol}\n"
                                f"Action: {trade_result['action'].upper()}\n"
                                f"Signal: {signal_type}\n"
                                f"OrderID: {trade_result.get('order_id', 'N/A')}"
                            )
                    elif trade_result['action'] != 'none':
                        print(f"[CTR Job] ⚠️ Trade skipped: {symbol} — {trade_result['details']}")
                        
                except Exception as e:
                    print(f"[CTR Job] ❌ Trade execution error: {e}")
            
        except Exception as e:
            print(f"[CTR Job] Signal callback error: {e}")
    
    def _save_signal(self, signal: Dict):
        """Зберегти сигнал в БД"""
        try:
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            
            sig_data = {
                'symbol': signal['symbol'],
                'type': signal['type'],
                'price': signal['price'],
                'stc': signal.get('stc', 0),
                'timeframe': signal.get('timeframe', self.timeframe if hasattr(self, 'timeframe') else '15m'),
                'smc_filtered': signal.get('smc_filtered', False),
                'reason': signal.get('reason', 'Crossover'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # FVG-specific fields
            if signal.get('is_fvg'):
                sig_data['is_fvg'] = True
                sig_data['sl_price'] = signal.get('sl_price', 0)
                sig_data['tp_price'] = signal.get('tp_price', 0)
                sig_data['fvg_high'] = signal.get('fvg_high', 0)
                sig_data['fvg_low'] = signal.get('fvg_low', 0)
                sig_data['rr_ratio'] = signal.get('rr_ratio', 0)
            
            signals.append(sig_data)
            
            # Зберігаємо останні 500 сигналів
            signals = signals[-500:]
            
            self.db.set_setting('ctr_signals', json.dumps(signals))
            
        except Exception as e:
            print(f"[CTR Job] Error saving signal: {e}")
    
    # ========================================
    # SL MONITOR
    # ========================================
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price from scanner's WebSocket cache"""
        if self._scanner and hasattr(self._scanner, '_cache'):
            cache = self._scanner._cache.get(symbol)
            if cache and cache.klines:
                return float(cache.klines[-1].close)
        return 0.0
    
    def _start_sl_monitor(self):
        """Start SL monitoring thread"""
        if self._sl_monitor_thread and self._sl_monitor_thread.is_alive():
            return
        
        self._sl_monitor_thread = threading.Thread(
            target=self._sl_monitor_loop, daemon=True, name="SL-Monitor"
        )
        self._sl_monitor_thread.start()
        print(f"[CTR Job] 🛑 SL Monitor started: {self.sl_monitor_pct}%, interval={self.sl_check_interval}s")
    
    def _sl_monitor_loop(self):
        """Main SL monitoring loop — checks every N seconds"""
        check_count = 0
        while self._running:
            try:
                if self.sl_monitor_enabled and self.sl_monitor_pct > 0:
                    self._check_sl_all_positions()
                    check_count += 1
                    # Log every 60 checks (~5 min at 5s interval)
                    if check_count % 60 == 1:
                        vp_count = len(self._virtual_positions)
                        sl_pct = float(self.db.get_setting('ctr_sl_monitor_pct', '0'))
                        print(f"[SL Monitor] ✅ Check #{check_count}: {vp_count} positions, SL={sl_pct}%")
            except Exception as e:
                print(f"[CTR Job] SL Monitor error: {e}")
            time.sleep(self.sl_check_interval)
    
    def _check_sl_all_positions(self):
        """Check all virtual positions for SL trigger"""
        now = time.time()
        sl_pct = float(self.db.get_setting('ctr_sl_monitor_pct', '0'))
        
        if sl_pct <= 0:
            return
        
        positions_copy = dict(self._virtual_positions)
        
        for symbol, pos in positions_copy.items():
            # Cooldown check — 60s after last SL
            last_sl = self._sl_cooldown.get(symbol, 0)
            if now - last_sl < 60:
                continue
            
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                continue
            
            entry_price = pos['entry_price']
            direction = pos['direction']
            
            # Calculate deviation
            if direction == 'BUY':  # LONG
                deviation_pct = (entry_price - current_price) / entry_price * 100
            else:  # SHORT
                deviation_pct = (current_price - entry_price) / entry_price * 100
            
            # Warning when approaching SL (>50% of threshold) — max once per 60s per symbol
            if deviation_pct > sl_pct * 0.5:
                warn_key = f"warn_{symbol}"
                last_warn = self._sl_cooldown.get(warn_key, 0)
                if now - last_warn >= 60:
                    label = 'LONG' if direction == 'BUY' else 'SHORT'
                    print(f"[SL Monitor] ⚠️ {symbol} {label}: deviation={deviation_pct:.2f}% / SL={sl_pct}% "
                          f"(entry=${entry_price:.4f}, now=${current_price:.4f})")
                    self._sl_cooldown[warn_key] = now
            
            # Check if SL triggered (deviation is loss percentage)
            if deviation_pct >= sl_pct:
                self._trigger_sl(symbol, direction, entry_price, current_price, deviation_pct, sl_pct)
                self._sl_cooldown[symbol] = now
    
    def _trigger_sl(self, symbol: str, old_direction: str, entry_price: float,
                    current_price: float, deviation_pct: float, sl_pct: float):
        """Trigger SL: create opposite signal"""
        new_direction = 'SELL' if old_direction == 'BUY' else 'BUY'
        old_label = 'LONG' if old_direction == 'BUY' else 'SHORT'
        new_label = 'SHORT' if old_direction == 'BUY' else 'LONG'
        
        print(f"[CTR Job] 🛑 SL TRIGGERED: {symbol} {old_label} → {new_label} "
              f"(deviation={deviation_pct:.2f}%, SL={sl_pct}%, "
              f"entry=${entry_price:.4f}, current=${current_price:.4f})")
        
        # 1. Create SL signal for Executed Signals table
        sl_signal = {
            'symbol': symbol,
            'type': new_direction,
            'price': current_price,
            'stc': 50.0,  # Neutral STC for SL signals
            'timeframe': self.timeframe if hasattr(self, 'timeframe') else '15m',
            'smc_filtered': False,
            'reason': f'🛑 SL ({sl_pct}%)',
            'is_sl': True,
            'sl_pct': sl_pct,
            'sl_deviation': round(deviation_pct, 2),
        }
        
        # Save to Executed Signals
        self._save_signal(sl_signal)
        
        # Update virtual position to new direction
        self._virtual_positions[symbol] = {
            'direction': new_direction,
            'entry_price': current_price,
            'timestamp': time.time()
        }
        
        # Update dedup direction
        self._last_signal_direction[symbol] = new_direction
        
        # 2. Send Telegram notification
        notifier = get_notifier()
        if notifier:
            pnl_direction = "📉" if old_direction == 'BUY' else "📈"
            msg = (f"🛑 STOP LOSS: {symbol}\n"
                   f"{pnl_direction} {old_label} закрито → {new_label} відкрито\n"
                   f"Entry: ${entry_price:.4f} → Current: ${current_price:.4f}\n"
                   f"Відхилення: -{deviation_pct:.2f}% (SL: {sl_pct}%)")
            notifier.send_message(msg)
        
        # 3. Execute on Bybit if Auto-Trade is ON
        if self._trade_executor:
            try:
                settings = self._trade_executor.get_settings()
                if settings['enabled'] and symbol in settings['trade_symbols']:
                    skip_native_sl = True  # Our SL is managing this
                    trade_result = self._trade_executor.execute_signal(
                        symbol=symbol,
                        signal_type=new_direction,
                        price=current_price,
                        reason=f'SL ({sl_pct}%)',
                        skip_native_sl=skip_native_sl
                    )
                    
                    if trade_result['success']:
                        print(f"[CTR Job] 💰 SL Trade executed: {symbol} {trade_result['action']}")
                        if notifier:
                            notifier.send_message(
                                f"💰 SL AUTO-TRADE: {symbol}\n"
                                f"Action: {trade_result['action'].upper()}\n"
                                f"Signal: {new_direction}\n"
                                f"OrderID: {trade_result.get('order_id', 'N/A')}"
                            )
                    else:
                        print(f"[CTR Job] ⚠️ SL Trade failed: {symbol} — {trade_result['details']}")
            except Exception as e:
                print(f"[CTR Job] ❌ SL Trade execution error: {e}")
    
    # ========================================
    # FVG DETECTOR
    # ========================================
    
    def _start_fvg_detector(self):
        """Initialize and start FVG detector using Bybit connector"""
        try:
            if self._fvg_detector:
                self._fvg_detector.stop()
            
            # Get Bybit connector from trade executor or create new one
            bybit = None
            if self._trade_executor and hasattr(self._trade_executor, 'bybit'):
                bybit = self._trade_executor.bybit
            else:
                try:
                    from core.bybit_connector import get_connector
                    bybit = get_connector()
                except:
                    print("[CTR Job] ⚠️ FVG: Cannot get Bybit connector")
                    return
            
            self._fvg_detector = FVGDetector(
                db=self.db,
                bybit_connector=bybit,
                timeframe=self.fvg_timeframe,
                min_fvg_pct=self.fvg_min_pct,
                max_fvg_per_symbol=self.fvg_max_per_symbol,
                rr_ratio=self.fvg_rr_ratio,
                sl_buffer_pct=self.fvg_sl_buffer_pct,
                scan_interval=self.fvg_scan_interval,
                on_signal=self._on_signal,
            )
            
            self._fvg_detector.start(
                watchlist=self.watchlist,
                price_getter=self._get_current_price
            )
        except Exception as e:
            print(f"[CTR Job] ❌ FVG Detector start error: {e}")
    
    def get_fvg_zones(self) -> List[Dict]:
        """Get FVG zones for UI"""
        if self._fvg_detector:
            return self._fvg_detector.get_zones()
        return []
    
    def get_fvg_stats(self) -> Dict:
        """Get FVG statistics"""
        if self._fvg_detector:
            return self._fvg_detector.get_stats()
        return {'running': False, 'active_fvg': 0, 'total_fvg': 0}
    
    def clear_fvg_zones(self) -> int:
        """Clear all FVG zones"""
        if self._fvg_detector:
            return self._fvg_detector.clear_zones()
        return 0
    
    def scan_fvg_now(self):
        """Manual FVG scan trigger"""
        if self._fvg_detector:
            self._fvg_detector.scan_now()
    
    def get_virtual_positions(self) -> List[Dict]:
        """Get all virtual positions with current P&L for UI"""
        result = []
        sl_pct = float(self.db.get_setting('ctr_sl_monitor_pct', '0'))
        
        for symbol, pos in self._virtual_positions.items():
            current_price = self._get_current_price(symbol)
            entry_price = pos['entry_price']
            direction = pos['direction']
            
            if current_price > 0 and entry_price > 0:
                if direction == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
            else:
                pnl_pct = 0
            
            result.append({
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': round(pnl_pct, 2),
                'sl_pct': sl_pct,
                'sl_distance': round(sl_pct - abs(min(pnl_pct, 0)), 2) if sl_pct > 0 and pnl_pct < 0 else sl_pct,
            })
        
        return sorted(result, key=lambda x: x['pnl_pct'])
    
    def _save_results(self):
        """Зберегти поточні результати в БД (включаючи SMC дані)"""
        if not self._scanner:
            return
        
        try:
            results = self._scanner.get_results()
            
            json_results = []
            for r in results:
                item = {
                    'symbol': r['symbol'],
                    'price': float(r['price']),
                    'stc': float(r['stc']),
                    'prev_stc': float(r['prev_stc']),
                    'status': r['status'],
                    'timeframe': r['timeframe']
                }
                
                # Додаємо SMC дані якщо є (per-symbol structure filter)
                smc = r.get('smc')
                if smc:
                    item['smc_trend'] = smc.get('trend', 'N/A')
                    item['smc_swing_high'] = smc.get('swing_high')
                    item['smc_swing_low'] = smc.get('swing_low')
                    item['smc_last_hh'] = smc.get('last_hh')
                    item['smc_last_hl'] = smc.get('last_hl')
                    item['smc_last_lh'] = smc.get('last_lh')
                    item['smc_last_ll'] = smc.get('last_ll')
                
                # HTF Trend (4h/1h) — overwrites smc_trend if available
                htf = r.get('smc_trend')
                if htf and isinstance(htf, dict):
                    item['smc_trend'] = htf  # {'4h': 'BULLISH', '1h': 'BEARISH'}
                
                json_results.append(item)
            
            self.db.set_setting('ctr_last_scan', json.dumps(json_results))
            self.db.set_setting('ctr_last_scan_time', datetime.now(timezone.utc).isoformat())
            
        except Exception as e:
            print(f"[CTR Job] Error saving results: {e}")
    
    def start(self) -> bool:
        """Запустити CTR сканер"""
        with self._lock:
            if self._running:
                print("[CTR Job] Already running")
                return True
            
            # Reload settings
            self._load_settings()
            
            if not self.watchlist:
                print("[CTR Job] ❌ Watchlist is empty")
                return False
            
            # Create scanner with filters
            self._scanner = CTRFastScanner(
                timeframe=self.timeframe,
                fast_length=self.fast_length,
                slow_length=self.slow_length,
                cycle_length=self.cycle_length,
                d1_length=self.d1_length,
                d2_length=self.d2_length,
                upper=self.upper,
                lower=self.lower,
                on_signal=self._on_signal,
                # Optional signal filters
                use_trend_guard=self.use_trend_guard,
                use_gap_detection=self.use_gap_detection,
                use_cooldown=self.use_cooldown,
                cooldown_seconds=self.cooldown_seconds,
                # SMC Filter
                smc_filter_enabled=self.smc_filter_enabled,
                smc_swing_length=self.smc_swing_length,
                smc_zone_threshold=self.smc_zone_threshold,
                smc_require_trend=self.smc_require_trend,
                # SMC Trend Filter (HTF)
                smc_trend_enabled=self.smc_trend_enabled,
                smc_trend_swing_4h=self.smc_trend_swing_4h,
                smc_trend_swing_1h=self.smc_trend_swing_1h,
                smc_trend_mode=self.smc_trend_mode,
                smc_trend_refresh=self.smc_trend_refresh,
                smc_trend_block_neutral=self.smc_trend_block_neutral,
                smc_trend_early_warning=self.smc_trend_early_warning,
                smc_trend_swing_15m=self.smc_trend_swing_15m,
            )
            
            # Start scanner (SMC Trend Filter is created internally by scanner)
            self._scanner.start(self.watchlist)
            self._running = True
            
            # Start results saver thread
            self._start_results_saver()
            
            # Start SL monitor if enabled
            if self.sl_monitor_enabled and self.sl_monitor_pct > 0:
                self._start_sl_monitor()
            
            # Start FVG Detector if enabled
            if self.fvg_enabled and FVG_AVAILABLE:
                self._start_fvg_detector()
            
            smc_status = "SMC✓" if self.smc_filter_enabled else ""
            sl_status = f" SL={self.sl_monitor_pct}%" if self.sl_monitor_enabled and self.sl_monitor_pct > 0 else ""
            fvg_status = f" FVG={self.fvg_timeframe}" if self.fvg_enabled and FVG_AVAILABLE else ""
            print(f"[CTR Job] ✅ Started with {len(self.watchlist)} symbols {smc_status}{sl_status}{fvg_status}")
            return True
    
    def stop(self):
        """Зупинити CTR сканер"""
        with self._lock:
            if not self._running:
                return
            
            if self._scanner:
                self._scanner.stop()
                self._scanner = None
            
            if self._fvg_detector:
                self._fvg_detector.stop()
                self._fvg_detector = None
            
            self._running = False
            print("[CTR Job] ❌ Stopped")
    
    def _start_results_saver(self):
        """Запустити потік для збереження результатів"""
        def saver_loop():
            import time
            while self._running:
                self._save_results()
                time.sleep(30)
        
        thread = threading.Thread(target=saver_loop, daemon=True)
        thread.start()
    
    def is_running(self) -> bool:
        return self._running
    
    def get_status(self) -> Dict:
        if self._scanner:
            return self._scanner.get_status()
        return {
            'running': False,
            'watchlist': self.watchlist,
            'timeframe': self.timeframe
        }
    
    def get_results(self) -> List[Dict]:
        if self._scanner:
            return self._scanner.get_results()
        return []
    
    def add_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        if self._scanner and self._running:
            return self._scanner.add_symbol(symbol)
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        if self._scanner and self._running:
            return self._scanner.remove_symbol(symbol)
        return True
    
    def reload_settings(self):
        """Перезавантажити налаштування"""
        self._load_settings()
        
        if self._scanner:
            self._scanner.reload_settings({
                'timeframe': self.timeframe,
                'upper': self.upper,
                'lower': self.lower,
                'fast_length': self.fast_length,
                'slow_length': self.slow_length,
                # Optional signal filters
                'use_trend_guard': self.use_trend_guard,
                'use_gap_detection': self.use_gap_detection,
                'use_cooldown': self.use_cooldown,
                'cooldown_seconds': self.cooldown_seconds,
                # SMC settings
                'smc_filter_enabled': self.smc_filter_enabled,
                'smc_swing_length': self.smc_swing_length,
                'smc_zone_threshold': self.smc_zone_threshold,
                'smc_require_trend': self.smc_require_trend,
                # SMC Trend Filter (HTF)
                'smc_trend_enabled': self.smc_trend_enabled,
                'smc_trend_swing_4h': self.smc_trend_swing_4h,
                'smc_trend_swing_1h': self.smc_trend_swing_1h,
                'smc_trend_mode': self.smc_trend_mode,
                'smc_trend_refresh': self.smc_trend_refresh,
                'smc_trend_block_neutral': self.smc_trend_block_neutral,
                'smc_trend_early_warning': self.smc_trend_early_warning,
                'smc_trend_swing_15m': self.smc_trend_swing_15m,
            })
        
        # Start/restart SL monitor if settings changed
        if self._running and self.sl_monitor_enabled and self.sl_monitor_pct > 0:
            if not (self._sl_monitor_thread and self._sl_monitor_thread.is_alive()):
                self._start_sl_monitor()
            print(f"[CTR Job] 🛑 SL Monitor: {self.sl_monitor_pct}%, interval={self.sl_check_interval}s")
        
        # FVG Detector reload
        if self._running and FVG_AVAILABLE:
            if self.fvg_enabled:
                if self._fvg_detector:
                    self._fvg_detector.reload_settings({
                        'min_fvg_pct': self.fvg_min_pct,
                        'max_fvg_per_symbol': self.fvg_max_per_symbol,
                        'rr_ratio': self.fvg_rr_ratio,
                        'sl_buffer_pct': self.fvg_sl_buffer_pct,
                        'timeframe': self.fvg_timeframe,
                    })
                else:
                    self._start_fvg_detector()
            elif self._fvg_detector:
                self._fvg_detector.stop()
                self._fvg_detector = None
                print("[CTR Job] 📐 FVG Detector disabled")
    
    def delete_signal(self, timestamp: str) -> bool:
        """Видалити конкретний сигнал за timestamp"""
        try:
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            
            original_len = len(signals)
            signals = [s for s in signals if s.get('timestamp') != timestamp]
            
            if len(signals) < original_len:
                self.db.set_setting('ctr_signals', json.dumps(signals))
                print(f"[CTR Job] 🗑️ Signal deleted (ts={timestamp[:19]})")
                return True
            
            return False
        except Exception as e:
            print(f"[CTR Job] Error deleting signal: {e}")
            return False
    
    def clear_signals(self) -> int:
        """Очистити всі сигнали"""
        try:
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            count = len(signals)
            
            self.db.set_setting('ctr_signals', '[]')
            self._last_signal_direction.clear()
            self._virtual_positions.clear()
            self._sl_cooldown.clear()
            
            print(f"[CTR Job] 🗑️ Cleared {count} signals + virtual positions + direction cache")
            return count
        except Exception as e:
            print(f"[CTR Job] Error clearing signals: {e}")
            return 0
    
    def reset_signal_direction(self, symbol: str):
        """Скинути останній напрямок для монети (дозволяє повторний сигнал)"""
        if symbol in self._last_signal_direction:
            old = self._last_signal_direction.pop(symbol)
            print(f"[CTR Job] 🔄 Reset direction for {symbol} (was {old})")
    
    def scan_now(self) -> List[Dict]:
        """Примусове сканування"""
        if self._scanner:
            results = self._scanner._scan_all()
            self._save_results()
            return results
        return []
    
    # =============================================
    # TRADE EXECUTOR ACCESS
    # =============================================
    
    def get_trade_executor(self) -> Optional['CTRTradeExecutor']:
        """Отримати торговий модуль"""
        return self._trade_executor
    
    def get_trade_settings(self) -> Dict:
        """Отримати налаштування торгівлі"""
        if self._trade_executor:
            return self._trade_executor.get_settings()
        return {
            'enabled': False,
            'leverage': 10,
            'deposit_pct': 5,
            'tp_pct': 0,
            'sl_pct': 0,
            'max_positions': 5,
            'trade_symbols': [],
        }
    
    def save_trade_settings(self, settings: Dict) -> bool:
        """Зберегти налаштування торгівлі"""
        if self._trade_executor:
            return self._trade_executor.save_settings(settings)
        return False
    
    def toggle_trade_symbol(self, symbol: str, enabled: bool) -> List[str]:
        """Додати/видалити символ з торгового списку"""
        if self._trade_executor:
            return self._trade_executor.set_trade_symbol(symbol, enabled)
        return []
    
    def get_trade_log(self, limit: int = 50) -> List[Dict]:
        """Отримати історію торгів"""
        if self._trade_executor:
            return self._trade_executor.get_trade_log(limit)
        return []
    
    def get_trade_status(self) -> Dict:
        """Повний статус торгівлі (позиції + баланс) — з кешем"""
        if not self._trade_executor:
            return {'available': False}
        
        return self._trade_executor.get_cached_status()
    
    # =============================================
    # SMC TREND FILTER ACCESS
    # =============================================
    
    def get_smc_trend_filter(self):
        """Отримати SMC Trend Filter (з scanner)"""
        if self._scanner:
            return self._scanner.get_smc_trend_filter()
        return None
    
    def get_smc_trend_status(self) -> Dict:
        """Повний статус SMC Trend Filter"""
        tf = self.get_smc_trend_filter()
        if tf:
            return tf.get_status()
        return {
            'enabled': False,
            'symbols_loaded': 0,
        }
    
    def get_smc_trend_for_symbol(self, symbol: str) -> Dict:
        """Тренд конкретного символу"""
        tf = self.get_smc_trend_filter()
        if tf and tf.enabled:
            return tf.get_symbol_trends(symbol)
        return {'4h': 'N/A', '1h': 'N/A'}


# ============================================
# SINGLETON & HELPERS
# ============================================

_ctr_job_instance: Optional[CTRFastJob] = None
_ctr_job_lock = threading.Lock()


def get_ctr_job(db: DBOperations = None) -> CTRFastJob:
    """Отримати singleton екземпляр CTR Job"""
    global _ctr_job_instance
    
    with _ctr_job_lock:
        if _ctr_job_instance is None:
            if db is None:
                from storage.db_operations import get_db
                db = get_db()
            _ctr_job_instance = CTRFastJob(db)
        return _ctr_job_instance


def start_ctr_job(db: DBOperations) -> CTRFastJob:
    """Запустити CTR Job"""
    job = get_ctr_job(db)
    job.start()
    return job


def stop_ctr_job():
    """Зупинити CTR Job"""
    global _ctr_job_instance
    
    with _ctr_job_lock:
        if _ctr_job_instance:
            _ctr_job_instance.stop()
