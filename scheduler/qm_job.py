"""
QM Zone Job v1.0 — Background Job для Quasimodo Zone Scanner

Завантажує налаштування з БД, керує сканером, зберігає сигнали.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from detection.qm_zone_scanner import QMZoneScanner, QMSignal
from alerts.telegram_notifier import get_notifier
from storage.db_operations import DBOperations


# ============================================
# ALL QM SETTINGS (з дефолтами)
# ============================================

QM_SETTINGS_DEFAULTS = {
    # HTF (SMC Zones)
    'qm_htf_timeframe': '15m',
    'qm_smc_swing_length': '50',
    'qm_smc_zone_threshold': '1.0',
    
    # LTF (QM Pattern)
    'qm_ltf_timeframe': '5m',
    'qm_min_swing_bars': '5',
    'qm_atr_period': '14',
    'qm_min_swing_atr': '1.2',
    'qm_min_pattern_bars': '25',
    'qm_lookback_bars': '150',
    'qm_min_db_diff_pct': '0.5',
    'qm_sl_buffer_pct': '0.2',
    'qm_min_confidence': '70',
    'qm_min_rr_ratio': '1.5',
    'qm_max_risk_pct': '2.0',
    
    # Scanner
    'qm_scan_interval': '15',
    'qm_enabled': '0',
}


class QMJob:
    """
    Background Job для QM Zone Scanner
    """
    
    def __init__(self, db: DBOperations):
        self.db = db
        self._scanner: Optional[QMZoneScanner] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Dedup: останній напрямок сигналу для кожного символу
        self._last_signal_direction: Dict[str, str] = {}
        
        # Load settings
        self._load_settings()
        self._load_last_directions()
    
    # ========================================
    # SETTINGS
    # ========================================
    
    def _load_settings(self):
        """Завантажити всі QM налаштування з БД"""
        def get(key):
            return self.db.get_setting(key, QM_SETTINGS_DEFAULTS.get(key, ''))
        
        # HTF
        self.htf_timeframe = get('qm_htf_timeframe')
        self.smc_swing_length = int(get('qm_smc_swing_length'))
        self.smc_zone_threshold = float(get('qm_smc_zone_threshold'))
        
        # LTF
        self.ltf_timeframe = get('qm_ltf_timeframe')
        self.qm_min_swing_bars = int(get('qm_min_swing_bars'))
        self.qm_atr_period = int(get('qm_atr_period'))
        self.qm_min_swing_atr = float(get('qm_min_swing_atr'))
        self.qm_min_pattern_bars = int(get('qm_min_pattern_bars'))
        self.qm_lookback_bars = int(get('qm_lookback_bars'))
        self.qm_min_db_diff_pct = float(get('qm_min_db_diff_pct'))
        self.qm_sl_buffer_pct = float(get('qm_sl_buffer_pct'))
        self.qm_min_confidence = float(get('qm_min_confidence'))
        self.qm_min_rr_ratio = float(get('qm_min_rr_ratio'))
        self.qm_max_risk_pct = float(get('qm_max_risk_pct'))
        
        # Scanner
        self.scan_interval = int(get('qm_scan_interval'))
        
        enabled_str = get('qm_enabled')
        self.enabled = enabled_str in ('1', 'true', 'True', 'yes')
        
        # Watchlist (спільна з CTR)
        watchlist_str = self.db.get_setting('qm_watchlist', '')
        if not watchlist_str:
            # Fallback: використовувати CTR watchlist
            watchlist_str = self.db.get_setting('ctr_watchlist', '')
        self.watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
    
    def _load_last_directions(self):
        """Завантажити останні напрямки для дедуплікації"""
        try:
            signals_str = self.db.get_setting('qm_signals', '[]')
            signals = json.loads(signals_str)
            for sig in signals:
                symbol = sig.get('symbol')
                direction = sig.get('direction')
                if symbol and direction:
                    self._last_signal_direction[symbol] = direction
            
            if self._last_signal_direction:
                print(f"[QM Job] 📋 Loaded directions for {len(self._last_signal_direction)} symbols")
        except Exception as e:
            print(f"[QM Job] Error loading directions: {e}")
    
    # ========================================
    # SIGNAL HANDLING
    # ========================================
    
    def _on_signal(self, signal: QMSignal):
        """Callback від сканера при знаходженні сигналу"""
        try:
            symbol = signal.symbol
            direction = signal.direction
            
            # Дедуплікація за напрямком
            last_dir = self._last_signal_direction.get(symbol)
            if last_dir == direction:
                print(f"[QM Job] ⏭️ Duplicate: {symbol} {direction} (same as last)")
                return
            
            # Оновити напрямок
            self._last_signal_direction[symbol] = direction
            
            # Відправити в Telegram
            notifier = get_notifier()
            if notifier:
                notifier.send_message(signal.format_telegram())
            
            # Зберегти в БД
            self._save_signal(signal)
            
            print(f"[QM Job] 📨 Signal sent: {symbol} {direction} "
                  f"(zone={signal.zone.level_name}, conf={signal.confidence:.0f}%)")
            
        except Exception as e:
            print(f"[QM Job] Signal error: {e}")
    
    def _save_signal(self, signal: QMSignal):
        """Зберегти сигнал в БД"""
        try:
            signals_str = self.db.get_setting('qm_signals', '[]')
            signals = json.loads(signals_str)
            
            signals.append(signal.to_dict())
            signals = signals[-100:]  # Тримаємо останні 100
            
            self.db.set_setting('qm_signals', json.dumps(signals))
        except Exception as e:
            print(f"[QM Job] Error saving signal: {e}")
    
    def _save_results(self):
        """Зберегти поточні результати сканування"""
        if not self._scanner:
            return
        
        try:
            results = self._scanner.get_results()
            self.db.set_setting('qm_last_scan', json.dumps(results))
            self.db.set_setting('qm_last_scan_time', datetime.now(timezone.utc).isoformat())
            
            status = self._scanner.get_status()
            self.db.set_setting('qm_status', json.dumps(status.get('stats', {})))
        except Exception as e:
            print(f"[QM Job] Save results error: {e}")
    
    # ========================================
    # LIFECYCLE
    # ========================================
    
    def start(self) -> bool:
        """Запустити QM Scanner"""
        with self._lock:
            if self._running:
                return True
            
            self._load_settings()
            
            if not self.watchlist:
                print("[QM Job] ❌ Empty watchlist")
                return False
            
            # Створити сканер
            self._scanner = QMZoneScanner(
                htf_timeframe=self.htf_timeframe,
                smc_swing_length=self.smc_swing_length,
                smc_zone_threshold=self.smc_zone_threshold,
                ltf_timeframe=self.ltf_timeframe,
                qm_min_swing_bars=self.qm_min_swing_bars,
                qm_atr_period=self.qm_atr_period,
                qm_min_swing_atr=self.qm_min_swing_atr,
                qm_min_pattern_bars=self.qm_min_pattern_bars,
                qm_lookback_bars=self.qm_lookback_bars,
                qm_min_db_diff_pct=self.qm_min_db_diff_pct,
                qm_sl_buffer_pct=self.qm_sl_buffer_pct,
                qm_min_confidence=self.qm_min_confidence,
                qm_min_rr_ratio=self.qm_min_rr_ratio,
                qm_max_risk_pct=self.qm_max_risk_pct,
                scan_interval=self.scan_interval,
                on_signal=self._on_signal,
            )
            
            self._scanner.start(self.watchlist)
            self._running = True
            
            # Фоновий потік для збереження результатів
            self._start_results_saver()
            
            print(f"[QM Job] ✅ Started: HTF={self.htf_timeframe}, LTF={self.ltf_timeframe}, "
                  f"{len(self.watchlist)} symbols")
            return True
    
    def stop(self):
        """Зупинити"""
        with self._lock:
            if not self._running:
                return
            if self._scanner:
                self._scanner.stop()
                self._scanner = None
            self._running = False
            print("[QM Job] ❌ Stopped")
    
    def _start_results_saver(self):
        """Потік для збереження результатів"""
        def saver():
            import time as _time
            while self._running:
                self._save_results()
                _time.sleep(30)
        
        t = threading.Thread(target=saver, daemon=True)
        t.start()
    
    def is_running(self) -> bool:
        return self._running
    
    def get_status(self) -> Dict:
        if self._scanner:
            return self._scanner.get_status()
        return {'running': False, 'watchlist': self.watchlist}
    
    def get_results(self) -> List[Dict]:
        if self._scanner:
            return self._scanner.get_results()
        return []
    
    def scan_now(self) -> List[Dict]:
        if self._scanner:
            results = self._scanner.scan_now()
            self._save_results()
            return results
        return []
    
    def add_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.db.set_setting('qm_watchlist', ','.join(self.watchlist))
        if self._scanner and self._running:
            return self._scanner.add_symbol(symbol)
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.db.set_setting('qm_watchlist', ','.join(self.watchlist))
        if self._scanner and self._running:
            self._scanner.remove_symbol(symbol)
        return True
    
    def reload_settings(self):
        """Hot reload налаштувань"""
        self._load_settings()
        if self._scanner:
            self._scanner.reload_settings({
                'htf_timeframe': self.htf_timeframe,
                'ltf_timeframe': self.ltf_timeframe,
                'smc_swing_length': self.smc_swing_length,
                'smc_zone_threshold': self.smc_zone_threshold,
                'scan_interval': self.scan_interval,
                'qm_min_swing_bars': self.qm_min_swing_bars,
                'qm_atr_period': self.qm_atr_period,
                'qm_min_swing_atr': self.qm_min_swing_atr,
                'qm_min_pattern_bars': self.qm_min_pattern_bars,
                'qm_lookback_bars': self.qm_lookback_bars,
                'qm_min_db_diff_pct': self.qm_min_db_diff_pct,
                'qm_sl_buffer_pct': self.qm_sl_buffer_pct,
                'qm_min_confidence': self.qm_min_confidence,
                'qm_min_rr_ratio': self.qm_min_rr_ratio,
                'qm_max_risk_pct': self.qm_max_risk_pct,
            })
    
    def delete_signal(self, timestamp: str) -> bool:
        """Видалити сигнал"""
        try:
            signals_str = self.db.get_setting('qm_signals', '[]')
            signals = json.loads(signals_str)
            original_len = len(signals)
            signals = [s for s in signals if s.get('timestamp') != timestamp]
            if len(signals) < original_len:
                self.db.set_setting('qm_signals', json.dumps(signals))
                return True
            return False
        except:
            return False
    
    def clear_signals(self) -> int:
        """Очистити всі сигнали"""
        try:
            signals_str = self.db.get_setting('qm_signals', '[]')
            count = len(json.loads(signals_str))
            self.db.set_setting('qm_signals', '[]')
            self._last_signal_direction.clear()
            return count
        except:
            return 0


# ============================================
# SINGLETON
# ============================================

_qm_job_instance: Optional[QMJob] = None
_qm_job_lock = threading.Lock()


def get_qm_job(db: DBOperations = None) -> QMJob:
    """Отримати singleton QM Job"""
    global _qm_job_instance
    with _qm_job_lock:
        if _qm_job_instance is None:
            if db is None:
                from storage.db_operations import get_db
                db = get_db()
            _qm_job_instance = QMJob(db)
        return _qm_job_instance


def start_qm_job(db: DBOperations) -> QMJob:
    job = get_qm_job(db)
    job.start()
    return job


def stop_qm_job():
    global _qm_job_instance
    with _qm_job_lock:
        if _qm_job_instance:
            _qm_job_instance.stop()
