"""
CTR Background Job v2.0 - Fast Edition

Інтеграція з CTRFastScanner для максимальної швидкості сигналів.
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
    - Автоматичне збереження результатів в БД
    """
    
    def __init__(self, db: DBOperations):
        self.db = db
        self._scanner: Optional[CTRFastScanner] = None
        self._running = False
        self._lock = threading.Lock()
        
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
        
        # Watchlist
        watchlist_str = self.db.get_setting('ctr_watchlist', '')
        self.watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
    
    def _on_signal(self, signal: Dict):
        """Callback при отриманні сигналу"""
        try:
            # Відправка в Telegram
            notifier = get_notifier()
            if notifier:
                notifier.send_message(signal['message'])
            
            # Збереження в БД
            self._save_signal(signal)
            
            print(f"[CTR Job] 📨 Signal sent: {signal['symbol']} {signal['type']}")
            
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
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Зберігаємо останні 100 сигналів
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
            
            # Конвертуємо для JSON
            json_results = []
            for r in results:
                json_results.append({
                    'symbol': r['symbol'],
                    'price': float(r['price']),
                    'stc': float(r['stc']),
                    'prev_stc': float(r['prev_stc']),
                    'status': r['status'],
                    'timeframe': r['timeframe']
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
            
            # Create scanner
            self._scanner = CTRFastScanner(
                timeframe=self.timeframe,
                fast_length=self.fast_length,
                slow_length=self.slow_length,
                cycle_length=self.cycle_length,
                d1_length=self.d1_length,
                d2_length=self.d2_length,
                upper=self.upper,
                lower=self.lower,
                on_signal=self._on_signal
            )
            
            # Start scanner
            self._scanner.start(self.watchlist)
            self._running = True
            
            # Start results saver thread
            self._start_results_saver()
            
            print(f"[CTR Job] ✅ Started with {len(self.watchlist)} symbols")
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
                time.sleep(30)  # Зберігаємо кожні 30 секунд
        
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
        
        # Оновити в БД
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        
        # Додати до сканера
        if self._scanner and self._running:
            return self._scanner.add_symbol(symbol)
        
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Видалити символ з watchlist"""
        symbol = symbol.upper()
        
        # Оновити в БД
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.db.set_setting('ctr_watchlist', ','.join(self.watchlist))
        
        # Видалити зі сканера
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
                'slow_length': self.slow_length
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
