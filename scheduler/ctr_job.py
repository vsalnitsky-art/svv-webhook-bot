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
from detection.zl_trend import ZeroLagTrendService
from strategy.zl_bot import ZLTBot
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
        self._zl_service: Optional[ZeroLagTrendService] = None
        self._zl_bot: Optional[ZLTBot] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Last signal direction per symbol for deduplication
        self._last_signal_direction: Dict[str, str] = {}  # symbol -> 'BUY'/'SELL'
        
        # Virtual positions for SL tracking (works without Auto-Trade)
        self._virtual_positions: Dict[str, Dict] = {}  # symbol -> {direction, entry_price, timestamp}
        self._sl_cooldown: Dict[str, float] = {}  # symbol -> timestamp of last SL trigger
        self._sl_monitor_thread: Optional[threading.Thread] = None
        
        # FVG TP Manager — track which FVG positions have been partially closed
        # symbol -> {entry_price, direction, partial_done: bool, be_set: bool}
        self._fvg_managed: Dict[str, Dict] = {}
        
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
        self.fvg_max_per_symbol = int(self.db.get_setting('ctr_fvg_max_per_symbol', '2'))
        self.fvg_rr_ratio = float(self.db.get_setting('ctr_fvg_rr_ratio', '1.5'))
        self.fvg_sl_buffer_pct = float(self.db.get_setting('ctr_fvg_sl_buffer_pct', '0.2'))
        self.fvg_scan_interval = int(self.db.get_setting('ctr_fvg_scan_interval', '60'))
        self.fvg_check_interval = int(self.db.get_setting('ctr_fvg_check_interval', '3'))
        self.fvg_trend_filter = _b('ctr_fvg_trend_filter', '0')
        self.fvg_trend_fast_ema = int(self.db.get_setting('ctr_fvg_trend_fast_ema', '5'))
        self.fvg_trend_slow_ema = int(self.db.get_setting('ctr_fvg_trend_slow_ema', '13'))
        self.fvg_htf_trend = _b('ctr_fvg_htf_trend', '0')
        self.fvg_htf_timeframe = self.db.get_setting('ctr_fvg_htf_timeframe', '1h')
        self.fvg_htf_fast_ema = int(self.db.get_setting('ctr_fvg_htf_fast_ema', '8'))
        self.fvg_htf_slow_ema = int(self.db.get_setting('ctr_fvg_htf_slow_ema', '21'))
        self.fvg_retest_enabled = _b('ctr_fvg_retest_enabled', '1')  # ON by default (current behavior)
        self.fvg_instant_enabled = _b('ctr_fvg_instant_enabled', '0')  # OFF by default (new)
        self.fvg_zl_trend_enabled = _b('ctr_fvg_zl_trend', '0')
        self.fvg_zl_15m_enabled = _b('ctr_fvg_zl_15m', '1')
        self.fvg_zl_1h_enabled = _b('ctr_fvg_zl_1h', '1')
        self.fvg_zl_4h_enabled = _b('ctr_fvg_zl_4h', '1')
        self.fvg_zl_length = int(self.db.get_setting('ctr_fvg_zl_length', '70'))
        self.fvg_zl_mult = float(self.db.get_setting('ctr_fvg_zl_mult', '1.2'))
        self.fvg_zl_5m_enabled = _b('ctr_fvg_zl_5m', '0')
        # ZLT Bot Strategy
        self.zl_bot_enabled = _b('ctr_zl_bot_enabled', '0')
        self.zl_bot_partial_pct = float(self.db.get_setting('ctr_zl_bot_partial_pct', '50'))
        
        # CTR Fast Scanner — enable/disable + EMA Trend Filter
        self.ctr_scanner_enabled = _b('ctr_scanner_enabled', '1')  # ON by default
        self.ctr_ema_trend_enabled = _b('ctr_ema_trend_enabled', '0')
        self.ctr_ema_trend_fast = int(self.db.get_setting('ctr_ema_trend_fast', '5'))
        self.ctr_ema_trend_slow = int(self.db.get_setting('ctr_ema_trend_slow', '13'))
        
        # FVG TP Manager — partial close + breakeven SL + trailing stop
        self.fvg_tp_enabled = _b('fvg_tp_manager_enabled', '0')
        self.fvg_tp_trigger_pct = float(self.db.get_setting('fvg_tp_trigger_pct', '0.5'))
        self.fvg_tp_close_pct = float(self.db.get_setting('fvg_tp_close_pct', '50'))
        self.fvg_tp_be_buffer_pct = float(self.db.get_setting('fvg_tp_be_buffer_pct', '0.05'))
        self.fvg_tp_trail_pct = float(self.db.get_setting('fvg_tp_trail_pct', '0.3'))
        self.fvg_tp_trail_start_pct = float(self.db.get_setting('fvg_tp_trail_start_pct', '0.8'))
        
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
    
    # ========================================
    # CTR EMA Trend Filter
    # ========================================
    
    @staticmethod
    def _compute_ema(closes, period: int) -> float:
        """Compute EMA from close prices (list or numpy array)."""
        if len(closes) < period:
            return 0.0
        k = 2.0 / (period + 1)
        ema = float(closes[0])
        for i in range(1, len(closes)):
            ema = float(closes[i]) * k + ema * (1 - k)
        return ema
    
    def _check_ctr_ema_trend(self, symbol: str, signal_type: str) -> bool:
        """
        EMA Trend Filter for CTR Fast signals.
        Fast EMA > Slow EMA → uptrend → BUY allowed
        Fast EMA < Slow EMA → downtrend → SELL allowed
        
        Returns True if signal is allowed, False if blocked.
        """
        if not self.ctr_ema_trend_enabled:
            return True
        
        if not self._scanner or not hasattr(self._scanner, '_cache'):
            return True  # no data, allow
        
        cache = self._scanner._cache.get(symbol)
        if not cache or not cache.klines:
            return True
        
        closes = [k.close for k in cache.klines]
        if len(closes) < self.ctr_ema_trend_slow + 5:
            return True
        
        fast_ema = self._compute_ema(closes, self.ctr_ema_trend_fast)
        slow_ema = self._compute_ema(closes, self.ctr_ema_trend_slow)
        
        if fast_ema == 0 or slow_ema == 0:
            return True
        
        is_uptrend = fast_ema > slow_ema
        
        if signal_type == 'BUY' and not is_uptrend:
            spread = (fast_ema - slow_ema) / slow_ema * 100
            print(f"[CTR Job] 🚫 EMA Trend BLOCKED: {symbol} BUY "
                  f"(trend=BEARISH, Fast={fast_ema:.4f}, Slow={slow_ema:.4f}, {spread:.2f}%)")
            return False
        
        if signal_type == 'SELL' and is_uptrend:
            spread = (fast_ema - slow_ema) / slow_ema * 100
            print(f"[CTR Job] 🚫 EMA Trend BLOCKED: {symbol} SELL "
                  f"(trend=BULLISH, Fast={fast_ema:.4f}, Slow={slow_ema:.4f}, +{spread:.2f}%)")
            return False
        
        return True
    
    def _on_signal(self, signal: Dict):
        """Callback при отриманні сигналу від сканера"""
        try:
            symbol = signal['symbol']
            signal_type = signal['type']
            reason = signal.get('reason', '')
            
            # === CTR FAST FILTERS (skip for FVG/SL signals) ===
            is_ctr_signal = not signal.get('is_fvg', False) and not signal.get('is_sl', False)
            
            if is_ctr_signal:
                # CTR Scanner enabled check
                if not self.ctr_scanner_enabled:
                    return  # silently skip — CTR signals disabled
                
                # EMA Trend Filter for CTR signals
                if not self._check_ctr_ema_trend(symbol, signal_type):
                    return  # blocked by trend
                
                # Zero Lag Trend Filter for CTR signals (shared service)
                if self._zl_service and self._zl_service.enabled:
                    direction = 'bullish' if signal_type == 'BUY' else 'bearish'
                    zl_ok, zl_block = self._zl_service.check_trend(symbol, direction)
                    if not zl_ok:
                        print(f"[CTR Job] 🚫 ZLT BLOCKED [{zl_block}]: {symbol} {signal_type}")
                        return
                
                # ZLT Bot conflict guard — don't trade against active bot position
                if self._zl_bot and self._zl_bot.enabled:
                    bot_state = self._zl_bot._states.get(symbol)
                    if bot_state and bot_state.state.value in ('in_trade', 'in_trade_partial'):
                        bot_dir = bot_state.direction  # 'LONG' or 'SHORT'
                        ctr_dir = signal_type  # 'BUY' or 'SELL'
                        # Block if CTR wants opposite of bot direction
                        if (bot_dir == 'LONG' and ctr_dir == 'SELL') or (bot_dir == 'SHORT' and ctr_dir == 'BUY'):
                            print(f"[CTR Job] 🚫 ZLT Bot CONFLICT: {symbol} CTR={ctr_dir} vs Bot={bot_dir} — skipped")
                            return
            
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
            
            # For FVG signals: track actual price after trade
            is_fvg = signal.get('is_fvg', False)
            fvg_actual_price = signal.get('price', 0)  # default to signal price
            fvg_trade_ok = False
            
            if send_telegram and notifier and not is_fvg:
                msg = signal.get('message', '')
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
            
            # FVG label for Telegram: "FVG Retest" or "FVG Instant" (from reason)
            fvg_label = "⚡ FVG Instant" if "Instant" in (reason or "") else "📐 FVG Retest"
            
            # === FVG DUPLICATE CHECK ===
            if is_fvg and self.fvg_tp_enabled and self._trade_executor and symbol in self._fvg_managed:
                existing = self._fvg_managed[symbol]
                if existing['direction'] == signal_type:
                    # Same direction — ignore duplicate
                    print(f"[FVG TP] ⏭️ {symbol}: ignoring {signal_type} — already managed in same direction")
                    # Still send Telegram signal notification
                    if send_telegram and notifier:
                        direction = '🟢 LONG' if signal_type == 'BUY' else '🔴 SHORT'
                        fvg_msg = (
                            f"{fvg_label} | {symbol} (duplicate, skip)\n"
                            f"{direction} @ ${signal.get('price', 0):.4f}\n"
                            f"FVG: ${signal.get('fvg_low', 0):.4f} – ${signal.get('fvg_high', 0):.4f}\n"
                            f"SL: ${signal.get('sl_price', 0):.4f} | "
                            f"TP: ${signal.get('tp_price', 0):.4f} | "
                            f"R:R {signal.get('rr_ratio', 1.5)}"
                        )
                        notifier.send_message(fvg_msg)
                    return
                else:
                    # Opposite direction — close existing fully, then open new
                    print(f"[FVG TP] 🔄 {symbol}: REVERSE — closing {existing['direction']} → opening {signal_type}")
                    try:
                        pos = self._trade_executor.get_position_for_symbol(symbol)
                        if pos and pos.get('size', 0) > 0:
                            close_result = self._trade_executor.bybit.close_position(
                                symbol, pos['side'], pos['size']
                            )
                            if close_result:
                                print(f"[FVG TP] ✅ {symbol}: closed {existing['direction']} fully (size={pos['size']})")
                                # Telegram
                                if notifier:
                                    old_dir = '🟢 LONG' if existing['direction'] == 'BUY' else '🔴 SHORT'
                                    new_dir = '🔴 SHORT' if existing['direction'] == 'BUY' else '🟢 LONG'
                                    notifier.send_message(
                                        f"📐 FVG Reverse | {symbol}\n"
                                        f"❌ Closed {old_dir} → Opening {new_dir}"
                                    )
                            else:
                                print(f"[FVG TP] ❌ {symbol}: failed to close existing position")
                        self._fvg_managed.pop(symbol, None)
                        time.sleep(0.3)  # Wait for position to clear
                    except Exception as e:
                        print(f"[FVG TP] ❌ {symbol}: reverse error: {e}")
                        self._fvg_managed.pop(symbol, None)
            
            # === AUTO-TRADE ===
            if self._trade_executor:
                try:
                    # When our SL monitor is active, tell executor to skip Bybit's native SL
                    skip_native_sl = self.sl_monitor_enabled and self.sl_monitor_pct > 0
                    
                    # FVG signals provide their own SL/TP
                    override_sl = signal.get('sl_price', 0) if is_fvg else 0
                    override_tp = signal.get('tp_price', 0) if is_fvg else 0
                    
                    # FVG always sets its own SL on exchange
                    if is_fvg and override_sl > 0:
                        skip_native_sl = False
                    
                    # FVG TP Manager: remove TP from exchange if we manage it ourselves
                    if is_fvg and self.fvg_tp_enabled:
                        override_tp = 0  # No exchange TP — we manage partial close
                    
                    # FVG without TP Manager: if Auto-Trade TP% is set, use it instead of R:R
                    if is_fvg and not self.fvg_tp_enabled:
                        at_settings = self._trade_executor.get_settings()
                        if at_settings.get('tp_pct', 0) > 0:
                            override_tp = 0  # Let _open_position calculate TP from tp_pct %
                    
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
                        fvg_trade_ok = True
                        trade_msg = f"[CTR Job] 💰 Trade executed: {symbol} {trade_result['action']} — {trade_result['details']}"
                        print(trade_msg)
                        
                        # Get actual entry price from exchange for FVG
                        if is_fvg:
                            try:
                                time.sleep(0.3)  # Wait for position to register
                                pos = self._trade_executor.get_position_for_symbol(symbol)
                                if pos and pos.get('entry_price', 0) > 0:
                                    fvg_actual_price = pos['entry_price']
                                    print(f"[CTR Job] 📐 FVG actual entry: {symbol} ${fvg_actual_price:.4f} "
                                          f"(signal was ${signal.get('price', 0):.4f})")
                            except Exception as e:
                                print(f"[CTR Job] ⚠️ Could not fetch actual price: {e}")
                        
                        # Register FVG position for TP manager
                        if is_fvg and self.fvg_tp_enabled:
                            self._fvg_managed[symbol] = {
                                'entry_price': fvg_actual_price,
                                'direction': signal_type,
                                'sl_price': signal.get('sl_price', 0),
                                'tp_price': signal.get('tp_price', 0),
                                'partial_done': False,
                                'be_set': False,
                                'timestamp': time.time(),
                            }
                            print(f"[CTR Job] 📐 FVG TP Manager: tracking {symbol} "
                                  f"{signal_type} @ ${fvg_actual_price:.4f}")
                        
                        # Non-FVG trade notification
                        if not is_fvg and notifier:
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
            
            # === FVG TELEGRAM (always, regardless of trade status) ===
            if is_fvg and send_telegram and notifier:
                direction = '🟢 LONG' if signal_type == 'BUY' else '🔴 SHORT'
                trade_tag = " ✅" if fvg_trade_ok else ""
                fvg_msg = (
                    f"{fvg_label} | {symbol}{trade_tag}\n"
                    f"{direction} @ ${fvg_actual_price:.4f}\n"
                    f"FVG: ${signal.get('fvg_low', 0):.4f} – ${signal.get('fvg_high', 0):.4f}\n"
                    f"SL: ${signal.get('sl_price', 0):.4f} | "
                    f"TP: ${signal.get('tp_price', 0):.4f} | "
                    f"R:R {signal.get('rr_ratio', 1.5)}"
                )
                notifier.send_message(fvg_msg)
            
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
            
            # ZLT Bot fields
            if signal.get('is_zlt_bot'):
                sig_data['is_zlt_bot'] = True
                sig_data['zlt_action'] = signal.get('zlt_action', '')
                sig_data['zlt_direction'] = signal.get('zlt_direction', '')
                sig_data['zlt_trends'] = signal.get('zlt_trends', '')
                sig_data['zlt_entry_price'] = signal.get('zlt_entry_price', 0)
            
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
        """Get current price from scanner's WebSocket cache, with Bybit REST fallback"""
        # Primary: WebSocket cache (fastest, <1ms)
        if self._scanner and hasattr(self._scanner, '_cache'):
            cache = self._scanner._cache.get(symbol)
            if cache and cache.klines:
                return float(cache.klines[-1].close)
        
        # Fallback: batch ticker cache (1 REST call for ALL symbols, refreshed every 30s)
        return self._get_fallback_price(symbol)
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Batch ticker fallback — single REST call cached for 30s"""
        now = time.time()
        
        # Refresh cache if stale (>30s) or empty
        if not hasattr(self, '_ticker_cache') or not self._ticker_cache:
            self._ticker_cache = {}
            self._ticker_cache_time = 0
        
        if now - self._ticker_cache_time > 30:
            try:
                tickers = self.bybit.get_tickers() if self.bybit else []
                if tickers:
                    self._ticker_cache = {
                        t['symbol']: float(t.get('lastPrice', 0))
                        for t in tickers if t.get('lastPrice')
                    }
                    self._ticker_cache_time = now
                    if not hasattr(self, '_ticker_fallback_logged'):
                        print(f"[CTR Job] 📡 Price fallback: Bybit tickers ({len(self._ticker_cache)} symbols)")
                        self._ticker_fallback_logged = True
            except Exception as e:
                pass
        
        return self._ticker_cache.get(symbol, 0.0)
    
    def _start_sl_monitor(self):
        """Start monitoring thread (SL Monitor + FVG TP + ZLT Bot)"""
        if self._sl_monitor_thread and self._sl_monitor_thread.is_alive():
            return
        
        self._sl_monitor_thread = threading.Thread(
            target=self._sl_monitor_loop, daemon=True, name="SL-Monitor"
        )
        self._sl_monitor_thread.start()
        
        parts = []
        if self.sl_monitor_enabled and self.sl_monitor_pct > 0:
            parts.append(f"SL={self.sl_monitor_pct}%")
        if self.fvg_tp_enabled:
            parts.append("FVG TP")
        if self.zl_bot_enabled:
            parts.append("ZLT Bot")
        label = ', '.join(parts) if parts else 'idle'
        print(f"[CTR Job] 🔄 Monitor started: [{label}], interval={self.sl_check_interval}s")
    
    def _sl_monitor_loop(self):
        """Main monitoring loop — SL Monitor + FVG TP Manager"""
        check_count = 0
        while self._running:
            try:
                # SL Monitor
                if self.sl_monitor_enabled and self.sl_monitor_pct > 0:
                    self._check_sl_all_positions()
                
                # FVG TP Manager
                if self.fvg_tp_enabled and self._fvg_managed:
                    self._check_fvg_tp_all()
                
                # ZLT Bot periodic safety check (every 6th cycle ~30s)
                if self._zl_bot and self._zl_bot.enabled and check_count % 6 == 0:
                    self._zl_bot.check_all()
                
                check_count += 1
                # Log every 60 checks (~5 min at 5s interval)
                if check_count % 60 == 1:
                    vp_count = len(self._virtual_positions)
                    fvg_count = len(self._fvg_managed)
                    sl_pct = float(self.db.get_setting('ctr_sl_monitor_pct', '0'))
                    parts = []
                    if self.sl_monitor_enabled:
                        parts.append(f"SL={sl_pct}%, {vp_count} virt.pos")
                    if self.fvg_tp_enabled:
                        trailing = sum(1 for v in self._fvg_managed.values() if v.get('trail_active'))
                        parts.append(f"FVG TP: {fvg_count} managed ({trailing} trailing)")
                    if parts:
                        print(f"[Monitor] ✅ Check #{check_count}: {', '.join(parts)}")
            except Exception as e:
                print(f"[CTR Job] Monitor error: {e}")
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
        
        # Remove from FVG TP manager if tracked
        self._fvg_managed.pop(symbol, None)
        
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
    # FVG TP MANAGER — Partial Close + BE SL + Trailing Stop
    # ========================================
    
    def _fetch_all_positions_map(self) -> Dict[str, Dict]:
        """Fetch ALL open positions in one REST call, return as symbol→position dict."""
        try:
            positions = self._trade_executor.bybit.get_positions()
            return {p['symbol']: p for p in positions} if positions else {}
        except Exception as e:
            print(f"[FVG TP] ❌ Batch positions error: {e}")
            return {}
    
    def _check_fvg_tp_all(self):
        """Check all FVG-managed positions: partial TP trigger + trailing stop."""
        if not self.fvg_tp_enabled or not self._trade_executor:
            return
        
        if not self._fvg_managed:
            return
        
        # 1 REST call for ALL positions (instead of N calls per symbol)
        all_positions = self._fetch_all_positions_map()
        
        managed_copy = dict(self._fvg_managed)
        
        for symbol, info in managed_copy.items():
            try:
                pos = all_positions.get(symbol)
                if not pos:
                    # Position closed externally — cleanup
                    self._fvg_managed.pop(symbol, None)
                    continue
                
                # Verify direction matches
                expected_side = "Buy" if info['direction'] == 'BUY' else "Sell"
                if pos['side'] != expected_side:
                    self._fvg_managed.pop(symbol, None)
                    continue
                
                entry_price = pos.get('entry_price', info['entry_price'])
                mark_price = pos.get('mark_price', 0)
                
                if mark_price <= 0 or entry_price <= 0:
                    continue
                
                # Calculate profit %
                if info['direction'] == 'BUY':
                    profit_pct = (mark_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - mark_price) / entry_price * 100
                
                if not info.get('partial_done'):
                    # Phase 1: waiting for partial TP trigger
                    if profit_pct >= self.fvg_tp_trigger_pct:
                        self._execute_fvg_partial_tp(symbol, info, pos, profit_pct)
                else:
                    # Phase 2: trailing stop on remaining position
                    self._update_fvg_trailing_sl(symbol, info, mark_price, profit_pct)
                    
            except Exception as e:
                print(f"[FVG TP] ❌ Error checking {symbol}: {e}")
    
    def _execute_fvg_partial_tp(self, symbol: str, info: Dict, pos: Dict, profit_pct: float):
        """Execute partial close + set breakeven SL for FVG position."""
        try:
            entry_price = pos.get('entry_price', info['entry_price'])
            size = pos.get('size', 0)
            direction = info['direction']
            label = 'LONG' if direction == 'BUY' else 'SHORT'
            
            # Calculate partial close qty
            close_qty = size * (self.fvg_tp_close_pct / 100)
            
            # Round qty to exchange specs
            specs = self._trade_executor._get_instrument_specs(symbol)
            if specs:
                close_qty = self._trade_executor._round_qty(close_qty, specs)
            
            if close_qty <= 0:
                print(f"[FVG TP] ⚠️ {symbol}: close_qty=0 after rounding, skip")
                return
            
            # 1. Partial close
            result = self._trade_executor.bybit.close_position(symbol, pos['side'], close_qty)
            
            if not result:
                print(f"[FVG TP] ❌ {symbol}: partial close failed")
                return
            
            print(f"[FVG TP] ✅ {symbol} {label}: closed {self.fvg_tp_close_pct}% "
                  f"(qty={close_qty}/{size}) at +{profit_pct:.2f}%")
            
            # 2. Calculate breakeven SL with buffer
            buffer_pct = self.fvg_tp_be_buffer_pct / 100
            if direction == 'BUY':
                be_sl = round(entry_price * (1 + buffer_pct), 8)
            else:
                be_sl = round(entry_price * (1 - buffer_pct), 8)
            
            # 3. Set breakeven SL on exchange
            sl_set = self._trade_executor.bybit.set_trading_stop(
                symbol=symbol,
                stop_loss=be_sl
            )
            
            if sl_set:
                print(f"[FVG TP] 🛡️ {symbol}: SL → breakeven ${be_sl:.4f} "
                      f"(entry ${entry_price:.4f} + {self.fvg_tp_be_buffer_pct}% buffer)")
            else:
                print(f"[FVG TP] ⚠️ {symbol}: failed to set BE SL")
            
            # 4. Mark as done + init trailing state
            mark_price = pos.get('mark_price', entry_price)
            info['partial_done'] = True
            info['be_set'] = sl_set
            info['be_price'] = be_sl
            info['current_sl'] = be_sl
            info['max_profit_price'] = mark_price  # Track peak price for trailing
            info['trail_active'] = False  # Trailing activates after trail_start_pct
            self._fvg_managed[symbol] = info
            
            # 5. Telegram notification
            notifier = get_notifier()
            if notifier:
                remaining = round(size - close_qty, 6)
                msg = (
                    f"📐 FVG TP Manager | {symbol}\n"
                    f"✅ Partial close: {self.fvg_tp_close_pct}% at +{profit_pct:.2f}%\n"
                    f"🛡️ SL → BE ${be_sl:.4f} ({label})\n"
                    f"📏 Trailing: {self.fvg_tp_trail_pct}% (activates at +{self.fvg_tp_trail_start_pct}%)\n"
                    f"Remaining: {remaining}"
                )
                notifier.send_message(msg)
            
            # 6. Log trade
            self._trade_executor._log_trade(
                symbol, 'PARTIAL_TP', 'Sell' if direction == 'BUY' else 'Buy',
                close_qty, entry_price, profit_pct, result.get('order_id', ''),
                leverage=0, margin=0
            )
            
        except Exception as e:
            print(f"[FVG TP] ❌ Execute error {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_fvg_trailing_sl(self, symbol: str, info: Dict, mark_price: float, profit_pct: float):
        """Update trailing stop for remaining FVG position after partial close."""
        direction = info['direction']
        entry_price = info['entry_price']
        trail_pct = self.fvg_tp_trail_pct / 100
        trail_start = self.fvg_tp_trail_start_pct
        
        max_price = info.get('max_profit_price', mark_price)
        
        # Update max profit price
        if direction == 'BUY' and mark_price > max_price:
            max_price = mark_price
            info['max_profit_price'] = max_price
        elif direction == 'SELL' and mark_price < max_price:
            max_price = mark_price
            info['max_profit_price'] = max_price
        
        # Activate trailing after trail_start_pct profit
        if not info.get('trail_active'):
            if profit_pct >= trail_start:
                info['trail_active'] = True
                print(f"[FVG TP] 📏 {symbol}: trailing activated at +{profit_pct:.2f}% "
                      f"(threshold: {trail_start}%)")
            else:
                self._fvg_managed[symbol] = info
                return
        
        # Calculate new trailing SL based on max profit price
        if direction == 'BUY':
            new_sl = round(max_price * (1 - trail_pct), 8)
        else:
            new_sl = round(max_price * (1 + trail_pct), 8)
        
        current_sl = info.get('current_sl', 0)
        
        # Only move SL in favorable direction (never lower for LONG, never higher for SHORT)
        should_update = False
        if direction == 'BUY' and new_sl > current_sl:
            should_update = True
        elif direction == 'SELL' and (current_sl <= 0 or new_sl < current_sl):
            should_update = True
        
        if should_update:
            sl_set = self._trade_executor.bybit.set_trading_stop(
                symbol=symbol,
                stop_loss=new_sl
            )
            
            if sl_set:
                label = 'LONG' if direction == 'BUY' else 'SHORT'
                print(f"[FVG TP] 📏 {symbol} {label}: trail SL ${current_sl:.4f} → ${new_sl:.4f} "
                      f"(peak=${max_price:.4f}, +{profit_pct:.2f}%)")
                info['current_sl'] = new_sl
            
        self._fvg_managed[symbol] = info
    
    # ========================================
    # FVG DETECTOR
    # ========================================
    
    def _start_zl_service(self, bybit):
        """Initialize Zero Lag Trend service + ZLT Bot."""
        try:
            if self._zl_service:
                self._zl_service.stop()
            
            # When ZLT Bot is enabled, force-enable all 4 TFs (bot needs them)
            zl_5m = self.fvg_zl_5m_enabled or self.zl_bot_enabled
            zl_15m = self.fvg_zl_15m_enabled or self.zl_bot_enabled
            zl_1h = self.fvg_zl_1h_enabled or self.zl_bot_enabled
            zl_4h = self.fvg_zl_4h_enabled or self.zl_bot_enabled
            
            self._zl_service = ZeroLagTrendService(
                bybit_connector=bybit,
                enabled=self.fvg_zl_trend_enabled or self.zl_bot_enabled,
                tf_5m_enabled=zl_5m,
                tf_15m_enabled=zl_15m,
                tf_1h_enabled=zl_1h,
                tf_4h_enabled=zl_4h,
                length=self.fvg_zl_length,
                mult=self.fvg_zl_mult,
            )
            
            if self._zl_service.enabled:
                # 30s when bot active (fastest 5m detection), 60s otherwise
                interval = 30 if self.zl_bot_enabled else 60
                self._zl_service.start(self.watchlist, update_interval=interval)
            
            # Create ZLT Bot
            if self.zl_bot_enabled and self._zl_service:
                self._start_zl_bot()
                
        except Exception as e:
            print(f"[CTR Job] ⚠️ ZLT service error: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_zl_bot(self):
        """Initialize ZLT Bot Strategy."""
        if not self._zl_service:
            return
        
        notifier = get_notifier()
        
        def _get_scanner_price(symbol: str) -> float:
            """Get current price from scanner cache."""
            if self._scanner and hasattr(self._scanner, '_cache'):
                cache = self._scanner._cache.get(symbol)
                if cache and cache.klines:
                    return cache.klines[-1].close
            return 0.0
        
        self._zl_bot = ZLTBot(
            zl_service=self._zl_service,
            enabled=True,
            partial_close_pct=self.zl_bot_partial_pct,
            exit_cooldown_sec=900,  # 15 min cooldown after full exit
            on_trade=self._on_zl_bot_trade,
            on_notify=notifier.send_message if notifier else None,
            get_price=_get_scanner_price,
        )
        self._zl_bot.set_watchlist(self.watchlist)
        
        # Do initial check for all symbols
        self._zl_bot.check_all()
        
        print(f"[ZLT Bot] ✅ Started: {len(self.watchlist)} symbols, "
              f"partial={self.zl_bot_partial_pct}%, cooldown=15min")
    
    def _on_zl_bot_trade(self, symbol: str, action: str, details: Dict):
        """Handle ZLT Bot trade actions — save signal + execute trade."""
        # Price comes from ZLT Bot (via get_price callback)
        current_price = details.get('price', 0)
        entry_price = details.get('entry_price', 0)
        
        # Get bot state for extra info
        bot_state = None
        if self._zl_bot and symbol in self._zl_bot._states:
            bot_state = self._zl_bot._states[symbol]
        
        # Get ZLT trends for signal context
        trends_str = ''
        if self._zl_service:
            trends = self._zl_service.get_all_trends(symbol)
            parts = []
            for tf_key, label in [('240', 'H4'), ('60', 'H1'), ('15', 'M15'), ('5', 'M5')]:
                t = trends.get(tf_key, '?')
                icon = '▲' if t == 'bullish' else '▼' if t == 'bearish' else '●'
                parts.append(f"{label}:{icon}")
            trends_str = ' '.join(parts)
        
        # === ALWAYS save to Executed Signals ===
        action_labels = {
            'entry': '⚡ ENTRY',
            'partial_exit': '🔻 PARTIAL',
            'reload': '🔄 RELOAD',
            'full_exit': '❌ EXIT',
        }
        
        signal_type = details.get('signal_type', 'BUY' if details.get('direction') == 'LONG' else 'SELL')
        reason_detail = details.get('reason', f'ZLT Bot {action}')
        
        sig = {
            'symbol': symbol,
            'type': signal_type,
            'price': current_price,
            'stc': 0,
            'timeframe': 'MTF',
            'smc_filtered': False,
            'reason': f"🤖 ZLT Bot {action_labels.get(action, action)} | {reason_detail}",
            'is_zlt_bot': True,
            'zlt_action': action,
            'zlt_direction': details.get('direction', ''),
            'zlt_trends': trends_str,
            'zlt_entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        # Add action-specific info
        if action == 'entry' and bot_state:
            sig['reason'] += f" (trade #{bot_state.trade_count})"
        elif action == 'partial_exit':
            pnl = ""
            if entry_price and current_price:
                d = details.get('direction', 'LONG')
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if d == 'LONG' else ((entry_price - current_price) / entry_price * 100)
                pnl = f" P&L:{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            sig['reason'] += f" ({details.get('close_pct', 50)}%){pnl}"
        elif action == 'reload':
            sig['reason'] += f" @ ${current_price:,.4f}" if current_price else ""
        elif action == 'full_exit':
            pnl = ""
            if entry_price and current_price:
                d = details.get('direction', 'LONG')
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if d == 'LONG' else ((entry_price - current_price) / entry_price * 100)
                pnl = f" P&L:{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            sig['reason'] += f" ({'partial→full' if details.get('was_partial') else 'full 100%'}){pnl}"
        
        self._save_signal(sig)
        
        # === Execute trade (if Auto-Trade enabled) ===
        if not self._trade_executor:
            print(f"[ZLT Bot] ⚠️ No trade executor for {symbol} {action}")
            return
        
        try:
            settings = self._trade_executor.get_settings()
            if not settings.get('enabled'):
                print(f"[ZLT Bot] ⚠️ Auto-Trade disabled — {symbol} {action} skipped")
                return
            if symbol not in settings.get('trade_symbols', []):
                print(f"[ZLT Bot] ⚠️ {symbol} not in trade symbols — {action} skipped")
                return
            
            if action == 'entry':
                result = self._trade_executor.execute_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=0,
                    reason=reason_detail,
                )
                if result['success']:
                    time.sleep(0.3)
                    pos = self._trade_executor.get_position_for_symbol(symbol)
                    entry_price = pos.get('entry_price', 0) if pos else 0
                    if bot_state:
                        bot_state.entry_price = entry_price
                    print(f"[ZLT Bot] 💰 Entry executed: {symbol} {signal_type} @ ${entry_price:.4f}")
                else:
                    print(f"[ZLT Bot] ⚠️ Entry failed: {symbol} — {result.get('details', '?')}")
            
            elif action == 'partial_exit':
                pos = self._trade_executor.get_position_for_symbol(symbol)
                if pos and pos.get('size', 0) > 0:
                    close_qty = round(pos['size'] * (details['close_pct'] / 100), 6)
                    if close_qty > 0:
                        result = self._trade_executor.bybit.close_position(
                            symbol, pos['side'], close_qty
                        )
                        if result:
                            print(f"[ZLT Bot] 💰 Partial close: {symbol} qty={close_qty}")
                        else:
                            print(f"[ZLT Bot] ⚠️ Partial close failed: {symbol}")
            
            elif action == 'reload':
                result = self._trade_executor.execute_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=0,
                    reason=reason_detail,
                )
                if result['success']:
                    print(f"[ZLT Bot] 💰 Reload executed: {symbol} {signal_type}")
            
            elif action == 'full_exit':
                pos = self._trade_executor.get_position_for_symbol(symbol)
                if pos and pos.get('size', 0) > 0:
                    result = self._trade_executor.bybit.close_position(
                        symbol, pos['side'], pos['size']
                    )
                    if result:
                        print(f"[ZLT Bot] 💰 Full exit: {symbol} size={pos['size']}")
                    else:
                        print(f"[ZLT Bot] ⚠️ Full exit failed: {symbol}")
                        
        except Exception as e:
            print(f"[ZLT Bot] ❌ Trade error {symbol} {action}: {e}")
    
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
            
            # Start ZLT service if not already running (may have been started in start())
            if not self._zl_service and self.fvg_zl_trend_enabled:
                self._start_zl_service(bybit)
            
            self._fvg_detector = FVGDetector(
                db=self.db,
                bybit_connector=bybit,
                timeframe=self.fvg_timeframe,
                min_fvg_pct=self.fvg_min_pct,
                max_fvg_per_symbol=self.fvg_max_per_symbol,
                rr_ratio=self.fvg_rr_ratio,
                sl_buffer_pct=self.fvg_sl_buffer_pct,
                scan_interval=self.fvg_scan_interval,
                check_interval=self.fvg_check_interval,
                trend_filter_enabled=self.fvg_trend_filter,
                trend_fast_ema=self.fvg_trend_fast_ema,
                trend_slow_ema=self.fvg_trend_slow_ema,
                htf_trend_enabled=self.fvg_htf_trend,
                htf_timeframe=self.fvg_htf_timeframe,
                htf_fast_ema=self.fvg_htf_fast_ema,
                htf_slow_ema=self.fvg_htf_slow_ema,
                retest_enabled=self.fvg_retest_enabled,
                instant_enabled=self.fvg_instant_enabled,
                zl_service=self._zl_service,
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
        """Get FVG statistics (includes ZLT data from shared service)"""
        if self._fvg_detector:
            return self._fvg_detector.get_stats()
        
        # FVG off — still return ZLT data for CTR UI
        stats = {'running': False, 'active_fvg': 0, 'total_fvg': 0}
        if self._zl_service:
            zl = self._zl_service
            stats.update({
                'zl_trend_enabled': zl.enabled,
                'zl_5m_enabled': zl.tf_5m_enabled,
                'zl_15m_enabled': zl.tf_15m_enabled,
                'zl_1h_enabled': zl.tf_1h_enabled,
                'zl_4h_enabled': zl.tf_4h_enabled,
                'zl_length': zl.length,
                'zl_mult': zl.mult,
                'zl_trends': zl._trends.copy(),
            })
        return stats
    
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
            
            # Start Zero Lag Trend service FIRST (shared by FVG + CTR + ZLT Bot)
            # Must compute trends before scanner starts generating signals
            if self.fvg_zl_trend_enabled or self.zl_bot_enabled:
                bybit = None
                if self._trade_executor and hasattr(self._trade_executor, 'bybit'):
                    bybit = self._trade_executor.bybit
                else:
                    try:
                        from core.bybit_connector import get_connector
                        bybit = get_connector()
                    except:
                        pass
                if bybit:
                    self._start_zl_service(bybit)
                    # Wait for initial ZLT calculation before starting scanner
                    print("[CTR Job] ⏳ Waiting for ZLT initial calculation...")
                    for _ in range(60):  # max 60s wait (10 symbols × ~3s each + REST)
                        if self._zl_service and self._zl_service._scan_counter > 0:
                            break
                        time.sleep(1)
                    trend_count = len(self._zl_service._trends) if self._zl_service else 0
                    print(f"[CTR Job] ✅ ZLT ready ({trend_count} symbols computed)")
            
            # Start scanner (SMC Trend Filter is created internally by scanner)
            self._scanner.signals_muted = not self.ctr_scanner_enabled
            self._scanner.start(self.watchlist)
            self._running = True
            
            # Start results saver thread
            self._start_results_saver()
            
            # Start monitoring thread if SL Monitor, FVG TP Manager, or ZLT Bot is enabled
            need_monitor = ((self.sl_monitor_enabled and self.sl_monitor_pct > 0) 
                           or self.fvg_tp_enabled or self.zl_bot_enabled)
            if need_monitor:
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
            
            if self._zl_service:
                self._zl_service.stop()
                self._zl_service = None
            
            if self._zl_bot:
                print("[ZLT Bot] ❌ Stopped")
                self._zl_bot = None
            
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
            results = self._scanner.get_results()
            # Enrich with ZLT trend data
            if self._zl_service and self._zl_service.enabled:
                for r in results:
                    sym = r.get('symbol', '')
                    zl_data = self._zl_service._trends.get(sym, {})
                    zl_summary = {}
                    for key, label in [('5', '5m'), ('15', '15m'), ('60', '1h'), ('240', '4h')]:
                        td = zl_data.get(key)
                        if td:
                            zl_summary[label] = td['trend'].upper()
                    r['zl_trend'] = zl_summary if zl_summary else None
            return results
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
            # Sync signal muting
            self._scanner.signals_muted = not self.ctr_scanner_enabled
        
        # Start/restart monitoring thread if needed
        need_monitor = ((self.sl_monitor_enabled and self.sl_monitor_pct > 0) 
                       or self.fvg_tp_enabled or self.zl_bot_enabled)
        if self._running and need_monitor:
            if not (self._sl_monitor_thread and self._sl_monitor_thread.is_alive()):
                self._start_sl_monitor()
        
        # ZLT Service + Bot dynamic start/stop
        if self._running:
            zlt_needed = self.fvg_zl_trend_enabled or self.zl_bot_enabled
            if zlt_needed and not self._zl_service:
                # Start ZLT service
                bybit = None
                if self._trade_executor and hasattr(self._trade_executor, 'bybit'):
                    bybit = self._trade_executor.bybit
                if bybit:
                    self._start_zl_service(bybit)
                    print("[CTR Job] 🔄 ZLT Service started (settings reload)")
            elif zlt_needed and self._zl_service:
                # Update existing service settings
                self._zl_service.update_settings(
                    enabled=True,
                    tf_5m_enabled=self.fvg_zl_5m_enabled or self.zl_bot_enabled,
                    tf_15m_enabled=self.fvg_zl_15m_enabled or self.zl_bot_enabled,
                    tf_1h_enabled=self.fvg_zl_1h_enabled or self.zl_bot_enabled,
                    tf_4h_enabled=self.fvg_zl_4h_enabled or self.zl_bot_enabled,
                )
            elif not zlt_needed and self._zl_service:
                self._zl_service.stop()
                self._zl_service = None
                print("[CTR Job] 🔄 ZLT Service stopped (settings reload)")
            
            # ZLT Bot
            if self.zl_bot_enabled and not self._zl_bot and self._zl_service:
                self._start_zl_bot()
                print("[CTR Job] 🔄 ZLT Bot started (settings reload)")
            elif self.zl_bot_enabled and self._zl_bot:
                self._zl_bot.enabled = True
                self._zl_bot.partial_close_pct = self.zl_bot_partial_pct
            elif not self.zl_bot_enabled and self._zl_bot:
                self._zl_bot.enabled = False
                print("[CTR Job] 🔄 ZLT Bot disabled (settings reload)")
        
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
                        'trend_filter_enabled': self.fvg_trend_filter,
                        'trend_fast_ema': self.fvg_trend_fast_ema,
                        'trend_slow_ema': self.fvg_trend_slow_ema,
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
            'sizing_mode': 'percent',
            'fixed_margin': 10,
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
