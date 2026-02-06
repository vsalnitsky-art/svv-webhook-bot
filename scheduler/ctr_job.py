"""
CTR Background Job v2.2 - Fast Edition + SMC Filter + Signal Deduplication

Інтеграція з CTRFastScanner для максимальної швидкості сигналів.
+ SMC Structure Filter для фільтрації сигналів.
+ Signal deduplication - не відправляє повторні сигнали в тому ж напрямку.
+ Zone detection - Premium/Discount/Equilibrium
"""

import threading
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from detection.ctr_scanner_fast import CTRFastScanner
from alerts.telegram_notifier import get_notifier
from storage.db_operations import DBOperations


class CTRFastJob:
    """
    Background job для CTR Fast Scanner
    
    Особливості:
    - Використовує WebSocket для real-time даних
    - Сигнали за 1-5 секунд
    - SMC Structure Filter для фільтрації
    - Signal deduplication - ігнорує повторні сигнали
    - Zone detection - показує зону ціни
    - Автоматичне збереження результатів в БД
    """
    
    def __init__(self, db: DBOperations):
        self.db = db
        self._scanner: Optional[CTRFastScanner] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Last signals cache for deduplication
        self._last_signals: Dict[str, str] = {}  # symbol -> last_signal_type
        
        # Load settings
        self._load_settings()
    
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
        
        # Watchlist - try DB first, fallback to settings
        try:
            watchlist = self._get_watchlist_from_db()
            if not watchlist:
                # Fallback to old setting
                watchlist_str = self.db.get_setting('ctr_watchlist', '')
                watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        except:
            watchlist_str = self.db.get_setting('ctr_watchlist', '')
            watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        self.watchlist = watchlist
        
        # Load last signals from DB for deduplication
        self._load_last_signals()
    
    def _get_watchlist_from_db(self) -> List[str]:
        """Get watchlist from CTR watchlist table"""
        try:
            return self.db.get_ctr_watchlist()
        except AttributeError:
            # Method doesn't exist yet
            return []
    
    def _load_last_signals(self):
        """Load last signal for each symbol from DB"""
        try:
            signals = self.db.get_ctr_signals(limit=100)
            for s in signals:
                symbol = s.get('symbol')
                if symbol and symbol not in self._last_signals:
                    self._last_signals[symbol] = s.get('type')
            print(f"[CTR Job] Loaded last signals for {len(self._last_signals)} symbols")
        except Exception as e:
            print(f"[CTR Job] Could not load last signals: {e}")
    
    def _is_duplicate_signal(self, symbol: str, signal_type: str) -> bool:
        """
        Check if this signal is a duplicate.
        Returns True if last signal for this symbol has the same direction.
        """
        last_type = self._last_signals.get(symbol)
        if last_type == signal_type:
            return True
        return False
    
    def _get_zone(self, price: float, smc_status: Dict) -> str:
        """
        Determine price zone: PREMIUM / DISCOUNT / EQUILIBRIUM
        Based on SMC trailing high/low
        """
        if not smc_status:
            return 'NEUTRAL'
        
        trailing_top = smc_status.get('trailing_top', 0)
        trailing_bottom = smc_status.get('trailing_bottom', 0)
        
        if not trailing_top or not trailing_bottom or trailing_top <= trailing_bottom:
            return 'NEUTRAL'
        
        range_size = trailing_top - trailing_bottom
        equilibrium = trailing_bottom + range_size * 0.5
        
        # Premium zone: above 50% (upper half)
        # Discount zone: below 50% (lower half)
        premium_threshold = trailing_bottom + range_size * 0.618  # ~61.8%
        discount_threshold = trailing_bottom + range_size * 0.382  # ~38.2%
        
        if price >= premium_threshold:
            return 'PREMIUM'
        elif price <= discount_threshold:
            return 'DISCOUNT'
        else:
            return 'EQUILIBRIUM'
    
    def _on_signal(self, signal: Dict):
        """Callback при отриманні сигналу"""
        try:
            symbol = signal['symbol']
            signal_type = signal['type']
            
            # Check for duplicate signal
            if self._is_duplicate_signal(symbol, signal_type):
                print(f"[CTR Job] ⏭️ Skipping duplicate signal: {symbol} {signal_type}")
                # Still save to DB but don't notify
                self._save_signal(signal, notified=False)
                return
            
            # Get zone if SMC enabled
            zone = 'NEUTRAL'
            smc_trend = None
            if self._scanner and self.smc_filter_enabled:
                smc_status = self._scanner.get_smc_status(symbol)
                if smc_status:
                    zone = self._get_zone(signal['price'], smc_status)
                    smc_trend = smc_status.get('trend_bias', 'NEUTRAL')
            
            # Update last signal cache
            self._last_signals[symbol] = signal_type
            
            # Build message with zone info
            zone_emoji = {'PREMIUM': '🔴', 'DISCOUNT': '🟢', 'EQUILIBRIUM': '⚪'}.get(zone, '⚪')
            zone_text = f" | Zone: {zone_emoji} {zone}" if zone != 'NEUTRAL' else ""
            
            message = signal['message']
            if zone_text and zone_text not in message:
                # Add zone info to message
                message = message.rstrip() + zone_text
            
            # Send notification
            notifier = get_notifier()
            if notifier:
                notifier.send_message(message)
            
            # Save to DB with additional info
            signal['zone'] = zone
            signal['smc_trend'] = smc_trend
            self._save_signal(signal, notified=True)
            
            smc_tag = " [SMC✓]" if signal.get('smc_filtered') else ""
            print(f"[CTR Job] 📨 Signal sent: {symbol} {signal_type}{smc_tag} | Zone: {zone}")
            
        except Exception as e:
            print(f"[CTR Job] Signal callback error: {e}")
    
    def _save_signal(self, signal: Dict, notified: bool = True):
        """Зберегти сигнал в БД"""
        try:
            # Try new DB method first
            try:
                self.db.add_ctr_signal(
                    symbol=signal['symbol'],
                    signal_type=signal['type'],
                    price=signal['price'],
                    stc=signal.get('stc'),
                    timeframe=signal.get('timeframe'),
                    smc_filtered=signal.get('smc_filtered', False),
                    smc_trend=signal.get('smc_trend'),
                    zone=signal.get('zone'),
                    notified=notified
                )
                return
            except AttributeError:
                pass
            
            # Fallback to old method (settings-based)
            signals_str = self.db.get_setting('ctr_signals', '[]')
            signals = json.loads(signals_str)
            
            signals.append({
                'symbol': signal['symbol'],
                'type': signal['type'],
                'price': signal['price'],
                'stc': signal.get('stc'),
                'timeframe': signal.get('timeframe'),
                'smc_filtered': signal.get('smc_filtered', False),
                'smc_trend': signal.get('smc_trend'),
                'zone': signal.get('zone'),
                'notified': notified,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Keep last 100 signals
            signals = signals[-100:]
            
            self.db.set_setting('ctr_signals', json.dumps(signals))
            
        except Exception as e:
            print(f"[CTR Job] Error saving signal: {e}")
    
    def _save_results(self):
        """Зберегти поточні результати в БД"""
        if not self._scanner:
            return
        
        try:
            results = self._scanner.get_results()
            
            # Add zone to results
            json_results = []
            for r in results:
                zone = 'NEUTRAL'
                smc_trend = None
                
                if self.smc_filter_enabled and r.get('smc'):
                    smc = r['smc']
                    zone = self._get_zone(r['price'], {
                        'trailing_top': smc.get('swing_high'),
                        'trailing_bottom': smc.get('swing_low')
                    })
                    smc_trend = smc.get('trend', 'NEUTRAL')
                
                json_results.append({
                    'symbol': r['symbol'],
                    'price': float(r['price']),
                    'stc': float(r['stc']),
                    'prev_stc': float(r['prev_stc']),
                    'status': r['status'],
                    'timeframe': r['timeframe'],
                    'zone': zone,
                    'smc_trend': smc_trend
                })
            
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
            
            # Create scanner with SMC filter
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
                # SMC Filter
                smc_filter_enabled=self.smc_filter_enabled,
                smc_swing_length=self.smc_swing_length,
                smc_zone_threshold=self.smc_zone_threshold,
            )
            
            # Start scanner
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
                time.sleep(30)  # Save every 30 seconds
        
        thread = threading.Thread(target=saver_loop, daemon=True)
        thread.start()
    
    def is_running(self) -> bool:
        """Перевірити чи працює сканер"""
        return self._running
    
    def get_status(self) -> Dict:
        """Отримати статус сканера"""
        if self._scanner:
            return self._scanner.get_status()
        return {
            'running': False,
            'watchlist': self.watchlist,
            'timeframe': self.timeframe
        }
    
    def get_results(self) -> List[Dict]:
        """Отримати результати сканування"""
        if self._scanner:
            return self._scanner.get_results()
        return []
    
    def add_symbol(self, symbol: str) -> bool:
        """Додати символ до watchlist"""
        symbol = symbol.upper()
        
        # Update in DB (new method)
        try:
            self.db.add_ctr_watchlist_item(symbol)
        except AttributeError:
            pass
        
        # Update in settings (old method - for compatibility)
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        
        # Add to scanner
        if self._scanner and self._running:
            return self._scanner.add_symbol(symbol)
        
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Видалити символ з watchlist та всі пов'язані дані"""
        symbol = symbol.upper()
        
        # Remove from DB with all data
        try:
            self.db.remove_ctr_watchlist_item(symbol, delete_data=True)
        except AttributeError:
            pass
        
        # Remove from settings (old method)
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        
        # Remove from last signals cache
        if symbol in self._last_signals:
            del self._last_signals[symbol]
        
        # Remove from scanner
        if self._scanner and self._running:
            return self._scanner.remove_symbol(symbol)
        
        return True
    
    def clear_signals(self, symbol: str = None) -> int:
        """Clear all signals or for specific symbol"""
        try:
            count = self.db.clear_ctr_signals(symbol)
            
            # Clear from cache
            if symbol:
                if symbol in self._last_signals:
                    del self._last_signals[symbol]
            else:
                self._last_signals.clear()
            
            return count
        except AttributeError:
            # Fallback to old method
            self.db.set_setting('ctr_signals', '[]')
            self._last_signals.clear()
            return 0
    
    def delete_signal(self, signal_id: int) -> bool:
        """Delete a specific signal"""
        try:
            return self.db.delete_ctr_signal(signal_id)
        except AttributeError:
            return False
    
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
                # SMC settings
                'smc_filter_enabled': self.smc_filter_enabled,
                'smc_swing_length': self.smc_swing_length,
                'smc_zone_threshold': self.smc_zone_threshold,
            })
    
    def scan_now(self) -> List[Dict]:
        """Примусове сканування"""
        if self._scanner:
            results = self._scanner._scan_all()
            self._save_results()
            return results
        return []


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
