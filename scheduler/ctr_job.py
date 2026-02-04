"""
CTR Background Job
==================

Фоновий процес для моніторингу CTR сигналів.
Сканує watchlist кожні N секунд і відправляє сигнали в Telegram.

v1.0 - Initial implementation
"""

import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Set

from detection.ctr_scanner import get_ctr_scanner
from alerts.telegram_notifier import get_notifier
from storage.db_operations import DatabaseOperations


class CTRBackgroundJob:
    """
    Background job for CTR signal monitoring
    """
    
    def __init__(self, db: DatabaseOperations):
        self.db = db
        self.scanner = get_ctr_scanner(db)
        self.notifier = get_notifier()
        
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Track sent signals to avoid duplicates
        self._sent_signals: Set[str] = set()
        
        # Scan interval (seconds)
        self.scan_interval = 30  # 30 seconds between scans
        
    def _get_watchlist(self) -> List[str]:
        """Get CTR watchlist from database"""
        try:
            watchlist = self.db.get_setting('ctr_watchlist', '')
            if not watchlist:
                return []
            
            # Parse comma-separated list
            symbols = [s.strip().upper() for s in watchlist.split(',') if s.strip()]
            return symbols
        except Exception as e:
            print(f"[CTR Job] Error getting watchlist: {e}")
            return []
    
    def _create_signal_key(self, signal: Dict) -> str:
        """Create unique key for signal to prevent duplicates"""
        symbol = signal['symbol']
        direction = 'BUY' if signal['buy_signal'] else 'SELL'
        # Include hour to allow same signal in different hours
        hour = datetime.now(timezone.utc).strftime('%Y%m%d%H')
        return f"{symbol}_{direction}_{hour}"
    
    def _send_signal(self, signal: Dict):
        """Send signal to Telegram"""
        signal_key = self._create_signal_key(signal)
        
        with self._lock:
            if signal_key in self._sent_signals:
                print(f"[CTR Job] Signal already sent: {signal_key}")
                return
            
            self._sent_signals.add(signal_key)
        
        # Format and send
        message = self.scanner.format_telegram_signal(signal)
        
        try:
            self.notifier.send_message(message)
            print(f"[CTR Job] ✅ Signal sent: {signal['symbol']} {'BUY' if signal['buy_signal'] else 'SELL'}")
            
            # Log to database
            self._log_signal(signal)
        except Exception as e:
            print(f"[CTR Job] ❌ Error sending signal: {e}")
    
    def _log_signal(self, signal: Dict):
        """Log signal to database"""
        try:
            direction = 'LONG' if signal['buy_signal'] else 'SHORT'
            self.db.log_event(
                message=f"CTR Signal: {signal['symbol']} {direction} @ ${signal['price']:.8f}",
                level='INFO',
                category='CTR_SIGNAL'
            )
        except Exception as e:
            print(f"[CTR Job] Error logging signal: {e}")
    
    def _cleanup_old_signals(self):
        """Remove old signal keys (older than 2 hours)"""
        current_hour = datetime.now(timezone.utc).strftime('%Y%m%d%H')
        prev_hour = (datetime.now(timezone.utc).replace(minute=0, second=0) 
                     - timedelta(hours=1)).strftime('%Y%m%d%H')
        
        with self._lock:
            self._sent_signals = {
                k for k in self._sent_signals 
                if k.endswith(current_hour) or k.endswith(prev_hour)
            }
    
    def _scan_loop(self):
        """Main scanning loop"""
        print("[CTR Job] Starting scan loop...")
        
        scan_count = 0
        
        while self._running:
            try:
                # Reload settings
                self.scanner.reload_settings()
                
                # Get watchlist
                watchlist = self._get_watchlist()
                
                if not watchlist:
                    print("[CTR Job] Watchlist empty, waiting...")
                    time.sleep(10)
                    continue
                
                scan_count += 1
                print(f"\n[CTR Job] Scan #{scan_count} - {len(watchlist)} symbols")
                
                # Scan all symbols
                results, signals = self.scanner.scan_watchlist(watchlist)
                
                # Log results
                for r in results:
                    status_emoji = "🟢" if r['status'] == 'Oversold' else "🔴" if r['status'] == 'Overbought' else "⚪"
                    print(f"  {status_emoji} {r['symbol']}: STC={r['stc']:.2f} ({r['status']})")
                
                # Send signals
                for signal in signals:
                    self._send_signal(signal)
                
                # Cleanup old signals every 10 scans
                if scan_count % 10 == 0:
                    self._cleanup_old_signals()
                
                # Store last scan results
                self._store_scan_results(results)
                
            except Exception as e:
                print(f"[CTR Job] Error in scan loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait for next scan
            time.sleep(self.scan_interval)
        
        print("[CTR Job] Scan loop stopped")
    
    def _store_scan_results(self, results: List[Dict]):
        """Store scan results in database for UI display"""
        try:
            import json
            self.db.set_setting('ctr_last_scan', json.dumps(results))
            self.db.set_setting('ctr_last_scan_time', datetime.now(timezone.utc).isoformat())
        except Exception as e:
            print(f"[CTR Job] Error storing results: {e}")
    
    def start(self):
        """Start background job"""
        if self._running:
            print("[CTR Job] Already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print("[CTR Job] ✅ Started")
    
    def stop(self):
        """Stop background job"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[CTR Job] ❌ Stopped")
    
    def is_running(self) -> bool:
        """Check if job is running"""
        return self._running


# Missing import
from datetime import timedelta

# Singleton
_job = None

def get_ctr_job(db: DatabaseOperations = None) -> CTRBackgroundJob:
    """Get CTR Background Job instance (singleton)"""
    global _job
    if _job is None and db is not None:
        _job = CTRBackgroundJob(db)
    return _job

def start_ctr_job(db: DatabaseOperations):
    """Start CTR background job"""
    job = get_ctr_job(db)
    job.start()
    return job

def stop_ctr_job():
    """Stop CTR background job"""
    global _job
    if _job:
        _job.stop()
