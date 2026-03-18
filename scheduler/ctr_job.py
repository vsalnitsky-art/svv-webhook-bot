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
        
        # Trade executor (Bybit Futures)
        self._trade_executor: Optional['CTRTradeExecutor'] = None
        if TRADE_AVAILABLE:
            self._init_trade_executor()
        
        # SMC Trend Filter (HTF)
        # Load settings
        self._load_settings()
        
        # Load last signal directions from DB for persistence across restarts
        self._load_last_directions()
    
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
            # because the scanner already handles the logic of when to fire them
            is_priority = "Trend Guard" in reason
            
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
                notifier.send_message(signal['message'])
            
            # Збереження в БД
            self._save_signal(signal)
            
            smc_tag = " [SMC✓]" if signal.get('smc_filtered') else ""
            reason_tag = f" [{reason}]" if reason else ""
            print(f"[CTR Job] 📨 Signal sent: {symbol} {signal_type}{smc_tag}{reason_tag}")
            
            # === AUTO-TRADE ===
            if self._trade_executor:
                try:
                    trade_result = self._trade_executor.execute_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=signal.get('price', 0),
                        reason=reason
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
            
            signals.append({
                'symbol': signal['symbol'],
                'type': signal['type'],
                'price': signal['price'],
                'stc': signal['stc'],
                'timeframe': signal['timeframe'],
                'smc_filtered': signal.get('smc_filtered', False),
                'reason': signal.get('reason', 'Crossover'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Зберігаємо останні 500 сигналів
            signals = signals[-500:]
            
            self.db.set_setting('ctr_signals', json.dumps(signals))
            
        except Exception as e:
            print(f"[CTR Job] Error saving signal: {e}")
    
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
            
            smc_status = "SMC✓" if self.smc_filter_enabled else ""
            print(f"[CTR Job] ✅ Started with {len(self.watchlist)} symbols {smc_status}")
            return True
    
    def stop(self):
        """Зупинити CTR сканер"""
        with self._lock:
            if not self._running:
                return
            
            if self._scanner:
                self._scanner.stop()
                self._scanner = None
            
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
            
            print(f"[CTR Job] 🗑️ Cleared {count} signals + direction cache")
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
