#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Utilities Module
Retry logic, structured logging, validation, metrics
"""
import logging
import json
import time
import functools
from datetime import datetime
from typing import Any, Callable, Optional
from enum import Enum
from decimal import Decimal

import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, RetryError, before_sleep_log
)

# ===== СТРУКТУРОВАНЕ ЛОГУВАННЯ =====
def setup_logging(log_level: str = 'INFO', format_type: str = 'json'):
    """Настройка структурованого логування з structlog"""
    
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    
    if format_type == 'json':
        # JSON логи для продакшну (легко парсити)
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Текстовий формат для розробки
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    logging.basicConfig(
        format='%(message)s',
        level=log_level_num,
        handlers=[logging.StreamHandler()]
    )

get_logger = structlog.get_logger

# ===== RETRY DECORATOR З EXPONENTIAL BACKOFF =====
def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator з exponential backoff для функцій API
    
    Usage:
        @with_retry(max_retries=3, exceptions=(RequestException,))
        def get_data(): ...
    """
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10) if backoff_factor else None,
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(get_logger(), logging.WARNING) if on_retry else None,
            reraise=True
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            try:
                return func(*args, **kwargs)
            except RetryError as e:
                logger.error("max_retries_exceeded", function=func.__name__, error=str(e))
                raise
        return wrapper
    return decorator

# ===== ВАЛІДАЦІЯ ВХІДНИХ ДАНИХ =====
class OrderSide(str, Enum):
    """Валідні сторони ордеру"""
    BUY = "Buy"
    SELL = "Sell"
    CLOSE = "Close"

class TradeDirection(str, Enum):
    """Напрямок торгу"""
    LONG = "Long"
    SHORT = "Short"

def validate_webhook_data(data: dict) -> dict:
    """
    Валідує дані з webhook-у від TradingView
    
    НОВИЙ ФОРМАТ (від TradingView):
    {
        "action": "Buy|Sell|Close",
        "symbol": "BTCUSDT" або "BTCUSDT.P",
        "direction": "Long|Short" (для Close),
        "riskPercent": float,
        "leverage": float,
        "entryPrice": float,
        "stopLossPercent": float (у відсотках!),
        "takeProfitPercent": float (у відсотках!),
        "filtersApproved": bool (опціонально)
    }
    
    NOTE: Символи з суфіксом ".P" автоматично нормалізуються (видаляється ".P")
    """
    errors = []
    
    # Перевірка action
    action = data.get('action', '').strip()
    if action not in ['Buy', 'Sell', 'Close']:
        errors.append(f"Invalid action: {action}. Must be 'Buy', 'Sell', or 'Close'")
    
    # Перевірка symbol та нормалізація (видалення ".P")
    symbol = data.get('symbol', '').strip()
    if not symbol or not isinstance(symbol, str):
        errors.append(f"Invalid symbol: {symbol}")
    else:
        # РІШЕННЯ: Видаляємо ".P" з символу якщо він присутній
        symbol = symbol.replace('.P', '')
        if not symbol.endswith(('USDT', 'BUSD')):
            errors.append(f"Symbol must end with USDT or BUSD: {symbol}")
    
    # Для Close - потребується direction
    if action == 'Close':
        direction = data.get('direction', '').strip()
        if direction not in ['Long', 'Short']:
            errors.append(f"For Close, direction must be 'Long' or 'Short', got: {direction}")
    
    # Перевірка risk
    if action in ['Buy', 'Sell']:
        risk = data.get('riskPercent')
        if risk is None:
            errors.append("riskPercent required for Buy/Sell")
        else:
            try:
                risk_val = float(risk)
                if not (0.1 <= risk_val <= 100):
                    errors.append(f"riskPercent must be 0.1-100%, got {risk_val}%")
            except (ValueError, TypeError):
                errors.append(f"riskPercent must be a number, got {risk}")
    
    # Перевірка leverage
    if action in ['Buy', 'Sell']:
        lev = data.get('leverage')
        if lev is None:
            errors.append("leverage required for Buy/Sell")
        else:
            try:
                lev_val = float(lev)
                if not (1 <= lev_val <= 125):
                    errors.append(f"leverage must be 1-125x, got {lev_val}x")
            except (ValueError, TypeError):
                errors.append(f"leverage must be a number, got {lev}")
    
    # Перевірка entryPrice (для Buy/Sell)
    if action in ['Buy', 'Sell']:
        entry_price = data.get('entryPrice')
        if entry_price is None:
            errors.append("entryPrice required for Buy/Sell")
        else:
            try:
                entry_val = float(entry_price)
                if entry_val <= 0:
                    errors.append(f"entryPrice must be > 0, got {entry_val}")
            except (ValueError, TypeError):
                errors.append(f"entryPrice must be a number, got {entry_price}")
    
    # Перевірка stopLossPercent (у відсотках!)
    sl_percent = data.get('stopLossPercent')
    if sl_percent is not None:
        try:
            sl_val = float(sl_percent)
            if not (0.01 <= sl_val <= 50):
                errors.append(f"stopLossPercent must be 0.01-50%, got {sl_val}%")
        except (ValueError, TypeError):
            errors.append(f"stopLossPercent must be a number, got {sl_percent}")
    
    # Перевірка takeProfitPercent (у відсотках!)
    # NOTE: TP=0 допускається і означає "використати налаштування бота"
    tp_percent = data.get('takeProfitPercent')
    if tp_percent is not None:
        try:
            tp_val = float(tp_percent)
            if tp_val != 0 and not (0.01 <= tp_val <= 100):
                errors.append(f"takeProfitPercent must be 0 (use bot settings) or 0.01-100%, got {tp_val}%")
        except (ValueError, TypeError):
            errors.append(f"takeProfitPercent must be a number, got {tp_percent}")
    
    if errors:
        raise ValueError("Webhook validation failed:\n" + "\n".join(errors))
    
    return {
        'action': action,
        'symbol': symbol,
        'direction': data.get('direction', ''),
        'riskPercent': float(data.get('riskPercent', 0)),
        'leverage': float(data.get('leverage', 0)),
        'entryPrice': float(data.get('entryPrice', 0)) if data.get('entryPrice') else None,
        'stopLossPercent': float(data.get('stopLossPercent')) if data.get('stopLossPercent') else None,
        'takeProfitPercent': float(data.get('takeProfitPercent')) if data.get('takeProfitPercent') is not None else None,
        'filtersApproved': data.get('filtersApproved', False)
    }

def validate_stop_loss(sl_price: float, entry_price: float, side: str) -> bool:
    """
    Валідує Stop Loss перед встановленням
    
    Args:
        sl_price: Ціна SL
        entry_price: Ціна входу
        side: 'Buy' або 'Sell'
    
    Returns:
        True якщо валідний, False інакше
    """
    if side == 'Buy':
        # Для лонга SL має бути НИЖЧЕ ціни входу
        return sl_price < entry_price and sl_price > 0
    elif side == 'Sell':
        # Для шорта SL має бути ВИЩЕ ціни входу
        return sl_price > entry_price
    return False

# ===== МЕТРИКИ =====
class TradeMetrics:
    """Простої метрики для моніторингу"""
    def __init__(self):
        self.trades_opened = 0
        self.trades_closed = 0
        self.trades_failed = 0
        self.total_pnl = 0.0
        self.start_time = datetime.utcnow()
    
    def log_trade_opened(self, symbol: str, qty: float, price: float):
        self.trades_opened += 1
        get_logger().info("trade_opened", symbol=symbol, qty=qty, price=price, total=self.trades_opened)
    
    def log_trade_closed(self, symbol: str, pnl: float):
        self.trades_closed += 1
        self.total_pnl += pnl
        get_logger().info("trade_closed", symbol=symbol, pnl=pnl, total_pnl=self.total_pnl)
    
    def log_trade_failed(self, symbol: str, reason: str):
        self.trades_failed += 1
        get_logger().error("trade_failed", symbol=symbol, reason=reason)
    
    def get_stats(self) -> dict:
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (self.trades_closed / (self.trades_opened or 1)) * 100
        return {
            'trades_opened': self.trades_opened,
            'trades_closed': self.trades_closed,
            'trades_failed': self.trades_failed,
            'total_pnl': round(self.total_pnl, 2),
            'success_rate': round(success_rate, 1),
            'uptime_seconds': int(uptime)
        }

metrics = TradeMetrics()

# ===== HELPERS =====
def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Безпечно конвертує значення у float.
    Bybit API може повертати '', None, або невалідні значення.
    """
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val: Any, default: int = 0) -> int:
    """
    Безпечно конвертує значення у int.
    Bybit API може повертати '', None, або невалідні значення.
    """
    if val is None or val == '':
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def price_to_decimal(price: float, decimals: int = 8) -> str:
    """Конвертує ціну в Decimal для точних розрахунків"""
    return str(round(Decimal(str(price)), decimals))
