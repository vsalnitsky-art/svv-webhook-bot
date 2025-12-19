#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ SCANNER COORDINATOR v1.0
============================
Координує роботу всіх сканерів в авторежимі.
Забезпечує почергове виконання без конфліктів.

Сканери:
- Smart Money OB Scanner
- Whale Hunter PRO
- Whale PRO
- Whale SNIPER
- RSI/MFI Screener

Автор: SVV Webhook Bot Team
"""

import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from settings_manager import settings

logger = logging.getLogger("ScannerCoordinator")


class ScannerType(Enum):
    SMART_MONEY = "smart_money"
    WHALE_HUNTER = "whale_hunter"
    WHALE_PRO = "whale_pro"
    WHALE_SNIPER = "whale_sniper"
    RSI_MFI = "rsi_mfi"


@dataclass
class ScannerInfo:
    """Інформація про сканер"""
    scanner_type: ScannerType
    name: str
    enabled_key: str  # Ключ в settings для auto_scan
    interval_key: str  # Ключ в settings для інтервалу
    scan_function: Optional[Callable] = None
    last_scan: Optional[datetime] = None
    is_scanning: bool = False
    priority: int = 0  # Менше = вищий пріоритет


class ScannerCoordinator:
    """
    🎛️ Координатор сканерів
    
    Забезпечує:
    - Почергове виконання сканерів
    - Уникнення конфліктів (тільки один сканер одночасно)
    - Динамічне додавання/видалення сканерів
    - Пріоритетну чергу
    """
    
    def __init__(self):
        self.scanners: Dict[ScannerType, ScannerInfo] = {}
        self.is_running = False
        self.current_scanner: Optional[ScannerType] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Базовий інтервал координатора (перевірка черги)
        self.check_interval = 5  # секунд
        
        # Реєструємо стандартні сканери
        self._register_default_scanners()
    
    def _register_default_scanners(self):
        """Реєструє стандартні сканери"""
        # Smart Money OB Scanner - найвищий пріоритет
        self.register_scanner(ScannerInfo(
            scanner_type=ScannerType.SMART_MONEY,
            name="Smart Money OB",
            enabled_key="ob_auto_scan",
            interval_key="ob_scan_interval",
            priority=1
        ))
        
        # RSI/MFI Screener
        self.register_scanner(ScannerInfo(
            scanner_type=ScannerType.RSI_MFI,
            name="RSI/MFI Screener",
            enabled_key="rsi_auto_scan",
            interval_key="rsi_scan_interval",
            priority=2
        ))
        
        # Whale Hunter PRO
        self.register_scanner(ScannerInfo(
            scanner_type=ScannerType.WHALE_HUNTER,
            name="Whale Hunter PRO",
            enabled_key="whp_auto_mode",
            interval_key="whp_auto_interval",
            priority=3
        ))
        
        # Whale PRO
        self.register_scanner(ScannerInfo(
            scanner_type=ScannerType.WHALE_PRO,
            name="Whale PRO",
            enabled_key="whale_pro_auto_scan",
            interval_key="whale_pro_scan_interval",
            priority=4
        ))
        
        # Whale SNIPER
        self.register_scanner(ScannerInfo(
            scanner_type=ScannerType.WHALE_SNIPER,
            name="Whale SNIPER",
            enabled_key="sniper_auto_scan",
            interval_key="sniper_scan_interval",
            priority=5
        ))
    
    def register_scanner(self, scanner_info: ScannerInfo):
        """Реєструє сканер"""
        self.scanners[scanner_info.scanner_type] = scanner_info
        logger.info(f"📝 Registered scanner: {scanner_info.name}")
    
    def set_scan_function(self, scanner_type: ScannerType, func: Callable):
        """Встановлює функцію сканування для сканера"""
        if scanner_type in self.scanners:
            self.scanners[scanner_type].scan_function = func
            logger.info(f"✅ Scan function set for: {self.scanners[scanner_type].name}")
    
    def is_scanner_enabled(self, scanner_type: ScannerType) -> bool:
        """Перевіряє чи сканер увімкнений"""
        if scanner_type not in self.scanners:
            return False
        
        scanner = self.scanners[scanner_type]
        return bool(settings.get(scanner.enabled_key, False))
    
    def get_scanner_interval(self, scanner_type: ScannerType) -> int:
        """Отримує інтервал сканера в секундах"""
        if scanner_type not in self.scanners:
            return 60
        
        scanner = self.scanners[scanner_type]
        interval = settings.get(scanner.interval_key, 60)
        
        # Конвертуємо хвилини в секунди якщо потрібно
        # Whale Hunter PRO використовує хвилини
        if scanner_type == ScannerType.WHALE_HUNTER:
            return int(interval) * 60
        
        return int(interval)
    
    def should_scan(self, scanner_type: ScannerType) -> bool:
        """Перевіряє чи потрібно запускати сканер"""
        if not self.is_scanner_enabled(scanner_type):
            return False
        
        scanner = self.scanners[scanner_type]
        
        if scanner.is_scanning:
            return False
        
        if scanner.last_scan is None:
            return True
        
        interval = self.get_scanner_interval(scanner_type)
        elapsed = (datetime.now() - scanner.last_scan).total_seconds()
        
        return elapsed >= interval
    
    def get_next_scanner(self) -> Optional[ScannerType]:
        """Отримує наступний сканер для виконання (по пріоритету)"""
        with self._lock:
            # Якщо хтось вже сканує - чекаємо
            if self.current_scanner is not None:
                return None
            
            # Сортуємо по пріоритету
            sorted_scanners = sorted(
                self.scanners.values(),
                key=lambda s: s.priority
            )
            
            for scanner in sorted_scanners:
                if self.should_scan(scanner.scanner_type):
                    return scanner.scanner_type
            
            return None
    
    def run_scanner(self, scanner_type: ScannerType) -> bool:
        """Запускає сканер"""
        if scanner_type not in self.scanners:
            return False
        
        scanner = self.scanners[scanner_type]
        
        if scanner.scan_function is None:
            logger.warning(f"⚠️ No scan function for: {scanner.name}")
            return False
        
        with self._lock:
            if self.current_scanner is not None:
                logger.warning(f"⚠️ Another scanner running: {self.current_scanner}")
                return False
            
            self.current_scanner = scanner_type
            scanner.is_scanning = True
        
        try:
            logger.info(f"🔍 Starting: {scanner.name}")
            scanner.scan_function()
            scanner.last_scan = datetime.now()
            logger.info(f"✅ Completed: {scanner.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in {scanner.name}: {e}")
            return False
            
        finally:
            with self._lock:
                scanner.is_scanning = False
                self.current_scanner = None
    
    def start(self):
        """Запускає координатор"""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        threading.Thread(target=self._coordinator_loop, daemon=True).start()
        logger.info("🎛️ Scanner Coordinator started")
    
    def stop(self):
        """Зупиняє координатор"""
        self.is_running = False
        self._stop_event.set()
        logger.info("Scanner Coordinator stopped")
    
    def _coordinator_loop(self):
        """Головний цикл координатора"""
        while not self._stop_event.is_set():
            try:
                # Шукаємо наступний сканер
                next_scanner = self.get_next_scanner()
                
                if next_scanner:
                    # Запускаємо в окремому потоці
                    threading.Thread(
                        target=self.run_scanner,
                        args=(next_scanner,),
                        daemon=True
                    ).start()
                
                # Чекаємо перед наступною перевіркою
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Coordinator error: {e}")
                time.sleep(10)
    
    def get_status(self) -> Dict:
        """Отримує статус всіх сканерів"""
        status = {
            'is_running': self.is_running,
            'current_scanner': self.current_scanner.value if self.current_scanner else None,
            'scanners': {}
        }
        
        for scanner_type, scanner in self.scanners.items():
            status['scanners'][scanner_type.value] = {
                'name': scanner.name,
                'enabled': self.is_scanner_enabled(scanner_type),
                'is_scanning': scanner.is_scanning,
                'interval': self.get_scanner_interval(scanner_type),
                'last_scan': scanner.last_scan.strftime('%H:%M:%S') if scanner.last_scan else None,
                'has_function': scanner.scan_function is not None
            }
        
        return status
    
    def trigger_scan(self, scanner_type: ScannerType) -> bool:
        """Примусово запускає сканер (обходить інтервал)"""
        if scanner_type not in self.scanners:
            return False
        
        if self.current_scanner is not None:
            logger.warning("Another scanner is running")
            return False
        
        threading.Thread(
            target=self.run_scanner,
            args=(scanner_type,),
            daemon=True
        ).start()
        
        return True


# Глобальний екземпляр
scanner_coordinator = ScannerCoordinator()


# ============================================================================
#                        WATCHLIST INTEGRATION
# ============================================================================

def add_to_smart_money_watchlist(
    symbol: str,
    direction: str,
    source: str
) -> Dict:
    """
    Додає монету до Smart Money Watchlist
    
    Використовується всіма сканерами для інтеграції.
    
    Args:
        symbol: Символ (BTCUSDT)
        direction: BUY або SELL
        source: Джерело (Whale Hunter PRO, RSI/MFI, etc.)
    
    Returns:
        Dict з результатом
    """
    from models import db_manager, SmartMoneyTicker
    from datetime import datetime
    
    session = db_manager.get_session()
    try:
        # Перевірка ліміту
        limit = int(settings.get('ob_watchlist_limit', 50))
        count = session.query(SmartMoneyTicker).count()
        
        if count >= limit:
            logger.warning(f"Watchlist limit reached ({limit})")
            return {'status': 'error', 'error': 'Watchlist limit reached'}
        
        # Перевірка дублікату
        existing = session.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
        if existing:
            logger.debug(f"Symbol already in watchlist: {symbol}")
            return {'status': 'exists', 'message': 'Already in watchlist'}
        
        # Додаємо
        item = SmartMoneyTicker(
            symbol=symbol.upper(),
            direction=direction,
            source=source,
            added_at=datetime.utcnow()
        )
        session.add(item)
        session.commit()
        
        logger.info(f"✅ Added to SM Watchlist: {symbol} {direction} from {source}")
        return {'status': 'ok', 'added_at': item.added_at.isoformat()}
        
    except Exception as e:
        session.rollback()
        logger.error(f"Add to watchlist error: {e}")
        return {'status': 'error', 'error': str(e)}
    finally:
        session.close()


def get_watchlist_symbols() -> List[str]:
    """Отримує список символів з Watchlist"""
    from models import db_manager, SmartMoneyTicker
    
    session = db_manager.get_session()
    try:
        items = session.query(SmartMoneyTicker.symbol).all()
        return [item.symbol for item in items]
    finally:
        session.close()
