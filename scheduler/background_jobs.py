"""
Background Jobs - фонові задачі для автоматизації
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config.bot_settings import DEFAULT_SETTINGS
# Import both v2 and v3 scanners
from detection.sleeper_scanner import get_sleeper_scanner
from detection.sleeper_scanner_v3 import get_sleeper_scanner_v3
from detection.ob_scanner import get_ob_scanner
from detection.signal_merger import get_signal_merger
from trading.position_tracker import get_position_tracker
from storage.db_operations import get_db
from alerts.telegram_notifier import get_notifier


class BackgroundJobs:
    """Менеджер фонових задач"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler(
            job_defaults={
                'coalesce': True,  # Об'єднувати пропущені запуски
                'max_instances': 1,  # Тільки один інстанс задачі
                'misfire_grace_time': 60  # Допуск на затримку
            }
        )
        self.is_running = False
        self.db = get_db()
        self.notifier = get_notifier()
        
        # Use v3 scanner by default (5-day strategy)
        self.use_v3_scanner = True
        
        # Статистика виконання
        self.job_stats: Dict[str, Dict[str, Any]] = {}
    
    def start(self):
        """Запустити scheduler"""
        if self.is_running:
            return
        
        self._setup_jobs()
        self.scheduler.start()
        self.is_running = True
        print("[SCHEDULER] Started background jobs")
        self.db.log_event(message="Background scheduler started", level="INFO", category="SYSTEM")
        
        # Run initial scan after 30 seconds to give time for app to fully start
        print("[SCHEDULER] Initial Sleeper scan scheduled in 30 seconds...")
        self.scheduler.add_job(
            self._job_sleeper_scan,
            'date',
            run_date=datetime.now() + timedelta(seconds=30),
            id='initial_sleeper_scan',
            name='Initial Sleeper Scan',
            replace_existing=True
        )
    
    def stop(self):
        """Зупинити scheduler"""
        if not self.is_running:
            return
        
        self.scheduler.shutdown(wait=False)
        self.is_running = False
        print("[SCHEDULER] Stopped background jobs")
        self.db.log_event(message="Background scheduler stopped", level="INFO", category="SYSTEM")
    
    def _setup_jobs(self):
        """Налаштувати всі фонові задачі"""
        
        # 1. Sleeper Scan - кожні 15 хвилин
        self.scheduler.add_job(
            self._job_sleeper_scan,
            IntervalTrigger(minutes=15),
            id='sleeper_scan',
            name='Sleeper Scanner',
            replace_existing=True
        )
        
        # 2. Order Block Scan для ready sleepers - кожні 5 хвилин
        self.scheduler.add_job(
            self._job_ob_scan,
            IntervalTrigger(minutes=5),
            id='ob_scan',
            name='Order Block Scanner',
            replace_existing=True
        )
        
        # 3. Signal Check - кожну хвилину
        self.scheduler.add_job(
            self._job_signal_check,
            IntervalTrigger(minutes=1),
            id='signal_check',
            name='Signal Checker',
            replace_existing=True
        )
        
        # 4. Position Monitor - кожні 30 секунд
        self.scheduler.add_job(
            self._job_position_monitor,
            IntervalTrigger(seconds=30),
            id='position_monitor',
            name='Position Monitor',
            replace_existing=True
        )
        
        # 5. Cleanup - кожну годину
        self.scheduler.add_job(
            self._job_cleanup,
            IntervalTrigger(hours=1),
            id='cleanup',
            name='Database Cleanup',
            replace_existing=True
        )
        
        # 6. Daily Summary - о 00:00
        self.scheduler.add_job(
            self._job_daily_summary,
            CronTrigger(hour=0, minute=0),
            id='daily_summary',
            name='Daily Summary',
            replace_existing=True
        )
        
        # 7. HP Update для sleepers - кожні 4 години
        self.scheduler.add_job(
            self._job_hp_update,
            IntervalTrigger(hours=4),
            id='hp_update',
            name='HP Update',
            replace_existing=True
        )
    
    def _log_job_execution(self, job_name: str, success: bool, duration: float, details: str = ""):
        """Логування виконання задачі"""
        if job_name not in self.job_stats:
            self.job_stats[job_name] = {
                'runs': 0,
                'successes': 0,
                'failures': 0,
                'total_duration': 0,
                'last_run': None,
                'last_status': None
            }
        
        stats = self.job_stats[job_name]
        stats['runs'] += 1
        stats['total_duration'] += duration
        stats['last_run'] = datetime.now()
        stats['last_status'] = 'success' if success else 'failed'
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        level = "INFO" if success else "ERROR"
        log_msg = f"[{job_name.upper()}] {details} ({duration:.2f}s)"
        
        # Console log for visibility
        print(f"[SCHEDULER] {'✓' if success else '✗'} {log_msg}")
        
        # Database log with correct argument order
        self.db.log_event(
            message=log_msg,
            level=level,
            category="SCHEDULER"
        )
    
    # ===== Job Implementations =====
    
    def _job_sleeper_scan(self):
        """Сканування Sleepers (5-Day Strategy v3)"""
        start = time.time()
        try:
            # Use v3 scanner (5-day strategy) by default
            if self.use_v3_scanner:
                scanner = get_sleeper_scanner_v3()
                results = scanner.run_scan()
            else:
                scanner = get_sleeper_scanner()
                results = scanner.scan()
            
            # Сповіщення про нові READY sleepers
            for sleeper in results:
                if sleeper.get('state') == 'READY':
                    # Check if just became ready (optional notification)
                    self.notifier.notify_sleeper_ready(sleeper)
            
            # Count by state
            states = {}
            for r in results:
                state = r.get('state', 'UNKNOWN')
                states[state] = states.get(state, 0) + 1
            
            duration = time.time() - start
            version = "v3 (5-Day)" if self.use_v3_scanner else "v2"
            self._log_job_execution(
                'sleeper_scan', 
                True, 
                duration,
                f"[{version}] {len(results)} candidates | READY:{states.get('READY', 0)} BUILD:{states.get('BUILDING', 0)} WATCH:{states.get('WATCHING', 0)}"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('sleeper_scan', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Sleeper scan: {e}")
    
    def _job_ob_scan(self):
        """Сканування Order Blocks для готових Sleepers"""
        start = time.time()
        try:
            # Отримати READY sleepers
            sleepers = self.db.get_sleepers(state='READY')
            if not sleepers:
                duration = time.time() - start
                self._log_job_execution('ob_scan', True, duration, "No ready sleepers")
                return
            
            scanner = get_ob_scanner()
            total_obs = 0
            
            for sleeper in sleepers:
                # sleeper is a dict
                obs = scanner.scan_symbol(sleeper['symbol'])
                total_obs += len(obs)
                
                # Сповіщення про якісні OB
                for ob in obs:
                    if ob.get('quality_score', 0) >= 70:
                        self.notifier.notify_ob_formed(ob)
            
            duration = time.time() - start
            self._log_job_execution(
                'ob_scan',
                True,
                duration,
                f"Found {total_obs} OBs for {len(sleepers)} sleepers"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('ob_scan', False, duration, str(e))
            print(f"[SCHEDULER ERROR] OB scan: {e}")
    
    def _job_signal_check(self):
        """Перевірка та обробка сигналів"""
        start = time.time()
        try:
            merger = get_signal_merger()
            signals = merger.check_for_signals()
            
            # Сповіщення про нові сигнали
            for signal in signals:
                self.notifier.notify_signal(signal)
            
            duration = time.time() - start
            self._log_job_execution(
                'signal_check',
                True,
                duration,
                f"Generated {len(signals)} signals"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('signal_check', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Signal check: {e}")
    
    def _job_position_monitor(self):
        """Моніторинг відкритих позицій"""
        start = time.time()
        try:
            tracker = get_position_tracker()
            
            # Перевірка TP/SL
            closed = tracker.check_tp_sl()
            
            # Сповіщення про закриті позиції
            for trade in closed:
                self.notifier.notify_trade_close(trade)
            
            # Оновлення trailing stops
            trailing_updates = tracker.update_all_trailing_stops()
            
            duration = time.time() - start
            self._log_job_execution(
                'position_monitor',
                True,
                duration,
                f"Closed {len(closed)}, trailing updates: {len(trailing_updates)}"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('position_monitor', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Position monitor: {e}")
    
    def _job_cleanup(self):
        """Очистка старих даних"""
        start = time.time()
        try:
            # Видалити мертві sleepers
            dead_count = self.db.remove_dead_sleepers()
            
            # Expire старі OB
            ob_scanner = get_ob_scanner()
            ob_scanner.cleanup_expired()
            
            # Очистити старі логи (>7 днів)
            self.db.clear_old_events(days=7)
            
            duration = time.time() - start
            self._log_job_execution(
                'cleanup',
                True,
                duration,
                f"Removed {dead_count} dead sleepers"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('cleanup', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Cleanup: {e}")
    
    def _job_daily_summary(self):
        """Денний звіт"""
        start = time.time()
        try:
            # Статистика за сьогодні
            today = datetime.now().date()
            stats = self.db.get_trade_stats(
                from_date=datetime.combine(today, datetime.min.time())
            )
            
            # Додаткова статистика
            sleepers = self.db.get_sleepers()
            ready_sleepers = [s for s in sleepers if s.state == 'READY']
            
            stats['sleepers_scanned'] = len(sleepers)
            stats['sleepers_ready'] = len(ready_sleepers)
            
            # Відправити в Telegram
            self.notifier.notify_daily_summary(stats)
            
            duration = time.time() - start
            self._log_job_execution('daily_summary', True, duration, "Sent")
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('daily_summary', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Daily summary: {e}")
    
    def _job_hp_update(self):
        """Оновлення HP для sleepers (integrated in v3 scan)"""
        start = time.time()
        try:
            # In v3, HP updates are handled during the scan
            # This job just triggers a full rescan
            if self.use_v3_scanner:
                scanner = get_sleeper_scanner_v3()
                results = scanner.run_scan()
                updated = len(results)
            else:
                scanner = get_sleeper_scanner()
                sleepers = self.db.get_sleepers()
                updated = 0
                for sleeper in sleepers:
                    result = scanner.check_single(sleeper['symbol'])
                    if result:
                        updated += 1
            
            sleepers = self.db.get_sleepers()
            
            duration = time.time() - start
            self._log_job_execution(
                'hp_update',
                True,
                duration,
                f"Updated {updated}/{len(sleepers)} sleepers"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('hp_update', False, duration, str(e))
            print(f"[SCHEDULER ERROR] HP update: {e}")
    
    # ===== Manual Triggers =====
    
    def trigger_job(self, job_id: str) -> bool:
        """Ручний запуск задачі"""
        job = self.scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now())
            return True
        return False
    
    def get_job_status(self) -> Dict[str, Any]:
        """Статус всіх задач"""
        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = {
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'stats': self.job_stats.get(job.id, {})
            }
            jobs.append(job_info)
        
        return {
            'is_running': self.is_running,
            'jobs': jobs
        }
    
    def pause_job(self, job_id: str) -> bool:
        """Призупинити задачу"""
        try:
            self.scheduler.pause_job(job_id)
            return True
        except:
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Відновити задачу"""
        try:
            self.scheduler.resume_job(job_id)
            return True
        except:
            return False


# ===== Singleton =====
_scheduler: Optional[BackgroundJobs] = None

def get_scheduler() -> BackgroundJobs:
    """Отримати singleton instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundJobs()
    return _scheduler
