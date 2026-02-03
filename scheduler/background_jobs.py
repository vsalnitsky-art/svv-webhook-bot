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
from detection.smc_signal_processor import get_smc_processor  # v8.1 SMC Entry
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
        
        # Load alert settings from DB
        self.notifier.load_alert_settings(self.db)
        
        # Use v3 scanner by default (5-day strategy)
        self.use_v3_scanner = True
        
        # Статистика виконання
        self.job_stats: Dict[str, Dict[str, Any]] = {}
        
        # v8.2.5: Lock для запобігання паралельним сканам
        self._scan_in_progress = False
        
        # ===== ANTI-SPAM: Cooldown tracking =====
        # Format: {'symbol': datetime_of_last_alert}
        self._sleeper_ready_sent: Dict[str, datetime] = {}  # Cooldown 2 hours
        self._intensive_alert_sent: Dict[str, datetime] = {}  # Cooldown 1 hour
        self._ob_alert_sent: Dict[str, datetime] = {}  # Cooldown 30 min
        
        # Track known READY sleepers - alert only on TRANSITION to READY
        self._known_ready_sleepers: set = set()  # Symbols currently in READY state
        
        # Cooldown periods (in seconds)
        self.COOLDOWN_SLEEPER_READY = 7200  # 2 hours - backup cooldown
        self.COOLDOWN_INTENSIVE = 3600  # 1 hour - don't spam same symbol
        self.COOLDOWN_OB = 1800  # 30 min - don't spam same OB
    
    # ===== MODULE CONTROL =====
    
    def _is_module_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled in settings
        
        Modules:
        - sleepers: Main sleeper scanner (enabled by default)
        - orderblocks: OB detection for ready sleepers
        - signals: Trade signal generation
        - positions: Position monitoring
        - intensive: Real-time monitoring of READY sleepers
        """
        setting_key = f'module_{module_name}'
        # Default enabled: sleepers
        default_enabled = ['sleepers']
        default_value = '1' if module_name in default_enabled else '0'
        value = self.db.get_setting(setting_key, default_value)
        return value in ('1', 'true', True, 1)
    
    # ===== ANTI-SPAM: Cooldown helpers =====
    
    def _can_send_alert(self, cooldown_dict: Dict[str, datetime], key: str, cooldown_seconds: int) -> bool:
        """Check if enough time passed since last alert for this key"""
        last_sent = cooldown_dict.get(key)
        if last_sent is None:
            return True
        
        elapsed = (datetime.now() - last_sent).total_seconds()
        return elapsed >= cooldown_seconds
    
    def _mark_alert_sent(self, cooldown_dict: Dict[str, datetime], key: str):
        """Mark that we sent an alert for this key"""
        cooldown_dict[key] = datetime.now()
    
    def _cleanup_old_cooldowns(self):
        """Remove old entries from cooldown dicts (run periodically)"""
        now = datetime.now()
        max_age = timedelta(hours=4)  # Keep entries for 4 hours max
        
        for d in [self._sleeper_ready_sent, self._intensive_alert_sent, self._ob_alert_sent]:
            keys_to_remove = [k for k, v in d.items() if (now - v) > max_age]
            for k in keys_to_remove:
                del d[k]
    
    def start(self):
        """Запустити scheduler"""
        if self.is_running:
            return
        
        self._setup_jobs()
        self.scheduler.start()
        self.is_running = True
        print("[SCHEDULER] Started background jobs")
        
        # Log enabled modules
        modules = ['sleepers', 'orderblocks', 'signals', 'positions', 'intensive']
        enabled = [m for m in modules if self._is_module_enabled(m)]
        disabled = [m for m in modules if not self._is_module_enabled(m)]
        print(f"[SCHEDULER] Enabled modules: {', '.join(enabled) if enabled else 'none'}")
        if disabled:
            print(f"[SCHEDULER] Disabled modules: {', '.join(disabled)}")
        
        self.db.log_event(
            message=f"Background scheduler started. Enabled: {', '.join(enabled)}", 
            level="INFO", 
            category="SYSTEM"
        )
        
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
        """
        Налаштувати всі фонові задачі
        
        v3.3: Optimized intervals to prevent Binance IP ban
        - Sleeper scan: 30 min (was 15)
        - OB scan: 10 min (was 5)
        - Signal check: 2 min (was 1)
        - Position monitor: 60 sec (was 30)
        """
        
        # 1. Sleeper Scan - кожні 30 хвилин (збільшено для захисту від бану)
        self.scheduler.add_job(
            self._job_sleeper_scan,
            IntervalTrigger(minutes=15),  # v8.2.4: Зменшено з 30 хв для швидшої детекції
            id='sleeper_scan',
            name='Sleeper Scanner',
            replace_existing=True
        )
        
        # 2. Order Block Scan для ready sleepers - кожні 10 хвилин
        self.scheduler.add_job(
            self._job_ob_scan,
            IntervalTrigger(minutes=10),
            id='ob_scan',
            name='Order Block Scanner',
            replace_existing=True
        )
        
        # 3. Signal Check - кожні 2 хвилини
        self.scheduler.add_job(
            self._job_signal_check,
            IntervalTrigger(minutes=2),
            id='signal_check',
            name='Signal Checker',
            replace_existing=True
        )
        
        # 4. Position Monitor - кожну хвилину
        self.scheduler.add_job(
            self._job_position_monitor,
            IntervalTrigger(seconds=60),
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
        
        # 8. Intensive READY Monitor - кожні 5 хвилин (v4.2)
        self.scheduler.add_job(
            self._job_intensive_ready_monitor,
            IntervalTrigger(minutes=5),
            id='intensive_ready_monitor',
            name='Intensive READY Monitor',
            replace_existing=True
        )
        
        # 9. SMC Entry Checker - кожні 5 хвилин (v8.1)
        # Перевіряє STALKING монети та шукає ENTRY_FOUND
        self.scheduler.add_job(
            self._job_smc_entry_check,
            IntervalTrigger(minutes=5),
            id='smc_entry_check',
            name='SMC Entry Checker',
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
        """Сканування Sleepers (SMC Strategy v8)"""
        # Check if module is enabled
        if not self._is_module_enabled('sleepers'):
            return
        
        # v8.2.5: Запобігаємо паралельним сканам
        if self._scan_in_progress:
            print("[SLEEPER] ⚠️ Scan already in progress, skipping...")
            return
        
        self._scan_in_progress = True
        start = time.time()
        try:
            # Use v3 scanner (5-day strategy) by default
            if self.use_v3_scanner:
                scanner = get_sleeper_scanner_v3()
                results = scanner.run_scan()
            else:
                scanner = get_sleeper_scanner()
                results = scanner.scan()
            
            # Track current READY sleepers and alert on TRANSITIONS
            current_ready = set()
            alerts_sent = 0
            
            for sleeper in results:
                symbol = sleeper.get('symbol', '')
                state = sleeper.get('state', '')
                
                if state == 'READY':
                    current_ready.add(symbol)
                    
                    # Alert only if this is a NEW ready (wasn't ready before)
                    if symbol not in self._known_ready_sleepers:
                        self.notifier.notify_sleeper_ready(sleeper)
                        alerts_sent += 1
                        print(f"[SLEEPER] 🔔 NEW READY: {symbol}")
            
            # Update known ready set
            # Sleepers that left READY will be removed, new READY will be added
            left_ready = self._known_ready_sleepers - current_ready
            if left_ready:
                print(f"[SLEEPER] Sleepers left READY: {left_ready}")
            
            self._known_ready_sleepers = current_ready
            
            # Periodic cleanup of old cooldowns
            self._cleanup_old_cooldowns()
            
            # Count by state
            states = {}
            for r in results:
                state = r.get('state', 'UNKNOWN')
                states[state] = states.get(state, 0) + 1
            
            duration = time.time() - start
            version = "v8.2.6 (UI Settings Integration)" if self.use_v3_scanner else "v2"
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
        finally:
            # v8.2.5: Завжди знімаємо lock
            self._scan_in_progress = False
    
    def _job_ob_scan(self):
        """Сканування Order Blocks для готових Sleepers"""
        # Check if module is enabled
        if not self._is_module_enabled('orderblocks'):
            return
        
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
            alerts_sent = 0
            
            for sleeper in sleepers:
                # sleeper is a dict
                obs = scanner.scan_symbol(sleeper['symbol'])
                total_obs += len(obs)
                
                # Сповіщення про якісні OB (з cooldown)
                for ob in obs:
                    if ob.get('quality_score', 0) >= 70:
                        # Cooldown key: symbol + direction
                        ob_key = f"{ob.get('symbol', '')}_{ob.get('direction', '')}"
                        if self._can_send_alert(self._ob_alert_sent, ob_key, self.COOLDOWN_OB):
                            self.notifier.notify_ob_formed(ob)
                            self._mark_alert_sent(self._ob_alert_sent, ob_key)
                            alerts_sent += 1
            
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
        # Check if module is enabled
        if not self._is_module_enabled('signals'):
            return
        
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
        # Check if module is enabled
        if not self._is_module_enabled('positions'):
            return
        
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
            # Статистика за сьогодні (days=1)
            stats = self.db.get_trade_stats(days=1)
            
            # Додаткова статистика
            sleepers = self.db.get_sleepers()
            ready_sleepers = [s for s in sleepers if s.get('state') == 'READY']
            
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
    
    def _job_intensive_ready_monitor(self):
        """
        Intensive monitoring for READY symbols - v4.2
        
        Runs every 5 minutes, checks:
        1. Volume spikes (>150% of average)
        2. Order book imbalance (>70%)
        3. Price momentum
        
        Sends HIGH priority alerts when conditions met
        """
        # Check if module is enabled
        if not self._is_module_enabled('intensive'):
            return
        
        start = time.time()
        try:
            from core.market_data import get_market_data
            
            # Get READY sleepers only
            ready_sleepers = self.db.get_sleepers(state='READY')
            
            if not ready_sleepers:
                duration = time.time() - start
                self._log_job_execution(
                    'intensive_ready_monitor',
                    True,
                    duration,
                    "No READY symbols to monitor"
                )
                return
            
            market = get_market_data()
            alerts_sent = 0
            
            for sleeper in ready_sleepers[:20]:  # Max 20 symbols
                symbol = sleeper['symbol']
                direction = sleeper.get('direction', 'NEUTRAL')
                
                try:
                    # Get recent data (minimal API calls)
                    ticker = market.get_ticker(symbol)
                    if not ticker:
                        continue
                    
                    # Check volume spike
                    volume_24h = float(ticker.get('quoteVolume', 0))
                    volume_ratio = sleeper.get('volume_ratio', 1.0)
                    
                    # Get current price and calculate momentum
                    current_price = float(ticker.get('lastPrice', 0))
                    price_change_pct = float(ticker.get('priceChangePercent', 0))
                    
                    # Alert conditions
                    is_volume_spike = volume_ratio and volume_ratio >= 1.5  # 150%+
                    is_strong_move = abs(price_change_pct) >= 2.0  # 2%+ move
                    is_direction_aligned = (
                        (direction == 'LONG' and price_change_pct > 0) or
                        (direction == 'SHORT' and price_change_pct < 0)
                    )
                    
                    # Determine alert priority
                    alert_priority = None
                    alert_reason = []
                    
                    # URGENT: Strong volume spike + price move + direction aligned
                    if volume_ratio and volume_ratio >= 2.5 and is_strong_move and is_direction_aligned:
                        alert_priority = 'URGENT'
                        alert_reason = ['Volume 250%+', f'Price {price_change_pct:+.1f}%', 'Direction OK']
                    
                    # HIGH: Volume spike + movement
                    elif is_volume_spike and (is_strong_move or is_direction_aligned):
                        alert_priority = 'HIGH'
                        if is_volume_spike:
                            alert_reason.append(f'Volume {volume_ratio:.1f}x')
                        if is_strong_move:
                            alert_reason.append(f'Price {price_change_pct:+.1f}%')
                    
                    # MEDIUM: Just volume spike on READY symbol
                    elif is_volume_spike:
                        alert_priority = 'MEDIUM'
                        alert_reason.append(f'Volume spike {volume_ratio:.1f}x')
                    
                    # Send alert if conditions met (with smart cooldown)
                    if alert_priority:
                        should_send = False
                        
                        # URGENT alerts ALWAYS go through - never miss breakouts!
                        if alert_priority == 'URGENT':
                            should_send = True
                        else:
                            # Check cooldown for HIGH/MEDIUM
                            # Key includes priority so higher priority can still fire
                            cooldown_key = f"{symbol}_{alert_priority}"
                            cooldown = 1800 if alert_priority == 'HIGH' else self.COOLDOWN_INTENSIVE  # 30m for HIGH, 1h for MEDIUM
                            
                            if self._can_send_alert(self._intensive_alert_sent, cooldown_key, cooldown):
                                should_send = True
                        
                        if should_send:
                            self._send_intensive_alert(
                                symbol=symbol,
                                direction=direction,
                                priority=alert_priority,
                                reasons=alert_reason,
                                sleeper=sleeper,
                                current_price=current_price,
                                price_change=price_change_pct,
                                volume_ratio=volume_ratio
                            )
                            
                            # Mark sent (for non-URGENT, track by symbol+priority)
                            if alert_priority != 'URGENT':
                                self._mark_alert_sent(self._intensive_alert_sent, f"{symbol}_{alert_priority}")
                            
                            alerts_sent += 1
                            
                            # Log event
                            self.db.log_event(
                                f"[INTENSIVE] {alert_priority} alert: {symbol} - {', '.join(alert_reason)}",
                                level='WARN' if alert_priority == 'URGENT' else 'INFO',
                                category='ALERT',
                                symbol=symbol
                            )
                
                except Exception as symbol_error:
                    print(f"[INTENSIVE] Error checking {symbol}: {symbol_error}")
                    continue
                
                # Small delay between symbols
                time.sleep(0.1)
            
            duration = time.time() - start
            self._log_job_execution(
                'intensive_ready_monitor',
                True,
                duration,
                f"Checked {len(ready_sleepers)} READY symbols, sent {alerts_sent} alerts"
            )
            
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('intensive_ready_monitor', False, duration, str(e))
            print(f"[SCHEDULER ERROR] Intensive monitor: {e}")
    
    def _send_intensive_alert(self, symbol: str, direction: str, priority: str,
                               reasons: list, sleeper: dict, current_price: float,
                               price_change: float, volume_ratio: float):
        """Send intensive monitoring alert via Telegram - v5 with phase info"""
        
        # Priority emojis
        priority_emoji = {
            'URGENT': '⚡⚡⚡',
            'HIGH': '🚀🔥',
            'MEDIUM': '👀📊'
        }
        
        direction_emoji = '🟢' if direction == 'LONG' else '🔴' if direction == 'SHORT' else '⚪'
        
        # v5: Phase info
        market_phase = sleeper.get('market_phase', 'UNKNOWN')
        phase_maturity = sleeper.get('phase_maturity', 'MIDDLE')
        is_reversal = sleeper.get('is_reversal_setup', False)
        exhaustion_score = sleeper.get('exhaustion_score', 0)
        
        # Phase emoji
        phase_emoji = {
            'ACCUMULATION': '📥',
            'MARKUP': '📈',
            'DISTRIBUTION': '📤',
            'MARKDOWN': '📉',
            'UNKNOWN': '❓'
        }.get(market_phase, '❓')
        
        # Maturity warning
        maturity_warning = ""
        if phase_maturity == 'EXHAUSTED':
            maturity_warning = "⚠️ EXHAUSTED - можливий розворот!"
        elif phase_maturity == 'LATE':
            maturity_warning = "⚠️ LATE phase - обережно!"
        
        # Reversal badge
        reversal_badge = "\n🔄 <b>REVERSAL SETUP!</b>" if is_reversal else ""
        
        # Format message
        message = f"""
{priority_emoji.get(priority, '📢')} <b>{priority} ALERT</b> {priority_emoji.get(priority, '📢')}
{direction_emoji} <b>{symbol}</b> | {direction}{reversal_badge}

📊 <b>Triggers:</b>
{chr(10).join(['• ' + r for r in reasons])}

💰 <b>Price:</b> {current_price:.4f} ({price_change:+.1f}%)
📈 <b>Volume:</b> {volume_ratio:.1f}x average
🎯 <b>Score:</b> {sleeper.get('total_score', 0):.0f}/100
❤️ <b>HP:</b> {sleeper.get('hp', 0)}/10

{phase_emoji} <b>Phase:</b> {market_phase} ({phase_maturity})
💀 <b>Exhaustion:</b> {exhaustion_score*100:.0f}%
{maturity_warning}

⏱ {datetime.now().strftime('%H:%M:%S UTC')}
"""
        
        # Check if intensive alerts are enabled
        if self.notifier.is_alert_enabled('intensive'):
            self.notifier.send_sync(message.strip())
    
    def _job_smc_entry_check(self):
        """
        v8.1: SMC Entry Checker - перевіряє STALKING/READY монети
        
        Логіка "Мисливця":
        1. Отримуємо всі READY та STALKING sleepers
        2. Для кожного перевіряємо: чи ціна вже відкотилась до OB?
        3. Якщо так → ENTRY_FOUND + Alert!
        4. Якщо timeout (>24h) → скасовуємо полювання
        """
        if not self._is_module_enabled('sleepers'):
            return
        
        start = time.time()
        
        try:
            # Отримуємо SMC Signal Processor
            smc_processor = get_smc_processor()
            
            # Обробляємо всі READY/STALKING sleepers
            results = smc_processor.process_ready_sleepers()
            
            # Підраховуємо статистику
            entries_found = sum(1 for r in results if r.action == "EXECUTE")
            stalking_count = smc_processor.get_stalking_count()
            
            duration = time.time() - start
            
            if entries_found > 0:
                symbols = [r.symbol for r in results if r.action == "EXECUTE"]
                self._log_job_execution(
                    'smc_entry_check',
                    True,
                    duration,
                    f"🎯 ENTRIES FOUND: {symbols} | Stalking: {stalking_count}"
                )
            else:
                self._log_job_execution(
                    'smc_entry_check',
                    True,
                    duration,
                    f"Stalking: {stalking_count} symbols"
                )
                
        except Exception as e:
            duration = time.time() - start
            self._log_job_execution('smc_entry_check', False, duration, str(e))
            print(f"[SCHEDULER ERROR] SMC Entry Check: {e}")
    
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
