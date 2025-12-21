#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SCANNER COORDINATOR v1.0                                    ║
║                                                                                ║
║  Централізована координація всіх сканерів бота                                ║
║  Запобігає перевантаженню API Bybit та конфліктам між модулями               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ScannerPriority(Enum):
    CRITICAL = 1    # Position Monitor
    HIGH = 2        # Confluence Scalper
    MEDIUM = 3      # Whale Hunter PRO, Smart Money
    LOW = 4         # OB Scanner, RSI Screener
    BACKGROUND = 5  # Analytics


class ScannerState(Enum):
    IDLE = "Idle"
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    ERROR = "Error"


@dataclass
class ScannerInfo:
    name: str
    priority: ScannerPriority
    state: ScannerState = ScannerState.IDLE
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    interval_seconds: int = 60
    api_calls_per_scan: int = 100
    is_enabled: bool = True
    callback: Optional[Callable] = None
    error_count: int = 0
    last_error: Optional[str] = None


class RateLimiter:
    MAX_REQUESTS_PER_SECOND = 50
    MAX_REQUESTS_PER_MINUTE = 1200
    
    def __init__(self):
        self.requests_this_second = 0
        self.requests_this_minute = 0
        self.last_second = datetime.now().second
        self.last_minute = datetime.now().minute
        self._lock = threading.Lock()
        self.request_history = deque(maxlen=1000)
    
    def can_make_request(self, count: int = 1) -> bool:
        with self._lock:
            now = datetime.now()
            if now.second != self.last_second:
                self.last_second = now.second
                self.requests_this_second = 0
            if now.minute != self.last_minute:
                self.last_minute = now.minute
                self.requests_this_minute = 0
            
            if self.requests_this_second + count > self.MAX_REQUESTS_PER_SECOND:
                return False
            if self.requests_this_minute + count > self.MAX_REQUESTS_PER_MINUTE:
                return False
            return True
    
    def record_requests(self, count: int = 1):
        with self._lock:
            self.requests_this_second += count
            self.requests_this_minute += count
            self.request_history.append({'time': datetime.now(), 'count': count})
    
    def wait_if_needed(self, count: int = 1):
        while not self.can_make_request(count):
            time.sleep(0.1)
        self.record_requests(count)
    
    def get_stats(self) -> Dict:
        return {
            'requests_this_second': self.requests_this_second,
            'requests_this_minute': self.requests_this_minute,
            'max_per_second': self.MAX_REQUESTS_PER_SECOND,
            'max_per_minute': self.MAX_REQUESTS_PER_MINUTE,
            'utilization_pct': round(self.requests_this_minute / self.MAX_REQUESTS_PER_MINUTE * 100, 1),
        }


class ScannerCoordinator:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.scanners: Dict[str, ScannerInfo] = {}
        self.rate_limiter = RateLimiter()
        self.running = False
        self._coordinator_thread = None
        self.current_scanner: Optional[str] = None
        
        self.stats = {
            'total_scans': 0,
            'scans_today': 0,
            'errors_today': 0,
            'api_calls_today': 0,
            'last_reset': datetime.now().strftime('%Y-%m-%d'),
        }
        
        self._register_default_scanners()
        logger.info("✅ Scanner Coordinator initialized")
    
    def _register_default_scanners(self):
        defaults = [
            ("position_monitor", ScannerPriority.CRITICAL, 5, 10),
            ("confluence_scalper", ScannerPriority.HIGH, 30, 150),
            ("whale_hunter_pro", ScannerPriority.MEDIUM, 60, 200),
            ("smart_money", ScannerPriority.MEDIUM, 120, 100),
            ("order_block_scanner", ScannerPriority.LOW, 180, 150),
            ("rsi_screener", ScannerPriority.LOW, 120, 100),
            ("analytics", ScannerPriority.BACKGROUND, 3600, 0),
        ]
        for name, priority, interval, api_calls in defaults:
            self.scanners[name] = ScannerInfo(
                name=name, priority=priority,
                interval_seconds=interval, api_calls_per_scan=api_calls
            )
    
    def register_callback(self, name: str, callback: Callable):
        if name in self.scanners:
            self.scanners[name].callback = callback
    
    def enable_scanner(self, name: str, enabled: bool = True):
        if name in self.scanners:
            self.scanners[name].is_enabled = enabled
    
    def set_interval(self, name: str, interval: int):
        if name in self.scanners:
            self.scanners[name].interval_seconds = interval
    
    def start(self):
        if self.running:
            return
        self.running = True
        self._coordinator_thread = threading.Thread(target=self._coordinator_loop, daemon=True)
        self._coordinator_thread.start()
        logger.info("🚀 Scanner Coordinator started")
    
    def stop(self):
        self.running = False
        logger.info("⏹️ Scanner Coordinator stopped")
    
    def _coordinator_loop(self):
        while self.running:
            try:
                self._update_stats()
                ready = self._get_ready_scanners()
                ready.sort(key=lambda x: x.priority.value)
                
                if ready:
                    self._execute_scanner(ready[0])
                time.sleep(1)
            except Exception as e:
                logger.error(f"Coordinator error: {e}")
                time.sleep(5)
    
    def _get_ready_scanners(self) -> List[ScannerInfo]:
        ready = []
        now = datetime.now()
        for scanner in self.scanners.values():
            if not scanner.is_enabled or scanner.state == ScannerState.RUNNING:
                continue
            if scanner.next_run is None or now >= scanner.next_run:
                if self.rate_limiter.can_make_request(scanner.api_calls_per_scan):
                    ready.append(scanner)
        return ready
    
    def _execute_scanner(self, scanner: ScannerInfo):
        try:
            scanner.state = ScannerState.RUNNING
            self.current_scanner = scanner.name
            
            logger.info(f"🔄 Running: {scanner.name}")
            self.rate_limiter.record_requests(scanner.api_calls_per_scan)
            self.stats['api_calls_today'] += scanner.api_calls_per_scan
            
            if scanner.callback:
                scanner.callback()
            else:
                self._run_module(scanner.name)
            
            scanner.state = ScannerState.IDLE
            scanner.last_run = datetime.now()
            scanner.next_run = datetime.now() + timedelta(seconds=scanner.interval_seconds)
            scanner.error_count = 0
            
            self.stats['total_scans'] += 1
            self.stats['scans_today'] += 1
            
        except Exception as e:
            scanner.state = ScannerState.ERROR
            scanner.error_count += 1
            scanner.last_error = str(e)
            self.stats['errors_today'] += 1
            delay = min(300, scanner.interval_seconds * (2 ** min(scanner.error_count, 5)))
            scanner.next_run = datetime.now() + timedelta(seconds=delay)
            logger.error(f"❌ {scanner.name} error: {e}")
        finally:
            self.current_scanner = None
    
    def _run_module(self, name: str):
        try:
            if name == "confluence_scalper":
                from confluence_scalper import confluence_scalper
                if confluence_scalper.auto_running and not confluence_scalper.is_scanning:
                    confluence_scalper.start_scan()
                    while confluence_scalper.is_scanning:
                        time.sleep(1)
            elif name == "position_monitor":
                from confluence_scalper import confluence_scalper
                if not confluence_scalper.monitor.running:
                    confluence_scalper.monitor.start()
            elif name == "whale_hunter_pro":
                try:
                    from whale_hunter_pro import whale_hunter_scanner
                    if whale_hunter_scanner.auto_running:
                        whale_hunter_scanner._run_single_scan()
                except ImportError:
                    pass
            elif name == "smart_money":
                try:
                    from smart_money_engine import smart_money_engine
                    if hasattr(smart_money_engine, 'auto_running') and smart_money_engine.auto_running:
                        smart_money_engine._scan_once()
                except ImportError:
                    pass
            elif name == "analytics":
                try:
                    from confluence_scalper import confluence_scalper
                    confluence_scalper.analytics.auto_adjust_settings()
                    confluence_scalper.analytics.update_blacklist()
                except:
                    pass
        except Exception as e:
            raise
    
    def _update_stats(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.stats['last_reset']:
            self.stats['scans_today'] = 0
            self.stats['errors_today'] = 0
            self.stats['api_calls_today'] = 0
            self.stats['last_reset'] = today
    
    def request_scan(self, name: str) -> bool:
        if name not in self.scanners:
            return False
        scanner = self.scanners[name]
        if not scanner.is_enabled or scanner.state == ScannerState.RUNNING:
            return False
        if not self.rate_limiter.can_make_request(scanner.api_calls_per_scan):
            return False
        scanner.next_run = datetime.now()
        return True
    
    def get_status(self) -> Dict:
        scanners_status = {}
        for name, s in self.scanners.items():
            scanners_status[name] = {
                'state': s.state.value,
                'priority': s.priority.name,
                'enabled': s.is_enabled,
                'interval': s.interval_seconds,
                'last_run': s.last_run.strftime('%H:%M:%S') if s.last_run else None,
                'next_run': s.next_run.strftime('%H:%M:%S') if s.next_run else None,
                'errors': s.error_count,
            }
        return {
            'running': self.running,
            'current': self.current_scanner,
            'scanners': scanners_status,
            'rate_limiter': self.rate_limiter.get_stats(),
            'stats': self.stats,
        }
    
    def pause_all(self):
        for s in self.scanners.values():
            if s.priority != ScannerPriority.CRITICAL:
                s.state = ScannerState.PAUSED
    
    def resume_all(self):
        for s in self.scanners.values():
            if s.state == ScannerState.PAUSED:
                s.state = ScannerState.IDLE


coordinator = ScannerCoordinator()


def register_routes(app):
    from flask import jsonify, request
    
    @app.route('/coordinator/status')
    def coord_status():
        return jsonify(coordinator.get_status())
    
    @app.route('/coordinator/start', methods=['POST'])
    def coord_start():
        coordinator.start()
        return jsonify({'status': 'started'})
    
    @app.route('/coordinator/stop', methods=['POST'])
    def coord_stop():
        coordinator.stop()
        return jsonify({'status': 'stopped'})
    
    @app.route('/coordinator/scanner/<name>/enable', methods=['POST'])
    def coord_enable(name):
        data = request.json or {}
        coordinator.enable_scanner(name, data.get('enabled', True))
        return jsonify({'status': 'ok'})
    
    @app.route('/coordinator/scanner/<name>/run', methods=['POST'])
    def coord_run(name):
        return jsonify({'status': 'ok' if coordinator.request_scan(name) else 'failed'})
    
    logger.info("✅ Scanner Coordinator routes registered")
