#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 SIGNAL FLIP SCALPER v1.0
===========================
Стратегія швидких угод на основі зміни сигналу RSI/MFI.

Концепція:
- Тригер: Сигнал змінюється BUY→SELL або SELL→BUY
- Фільтри: RSI Zone, MFI, Volume, BTC Trend, HTF, ATR, Time, Momentum
- Всі фільтри опціональні - працює при вимкненні будь-якого

Timeframe Presets (оптимізовано для Bybit):
- 1m:  Ultra Scalping (TP 0.15%, SL 0.2%, Hold 5min)
- 5m:  Fast Scalping (TP 0.25%, SL 0.3%, Hold 15min)
- 15m: Standard Scalping (TP 0.4%, SL 0.5%, Hold 45min) [DEFAULT]
- 30m: Slow Scalping (TP 0.6%, SL 0.7%, Hold 90min)
- 1h:  Swing Scalping (TP 1.0%, SL 1.0%, Hold 4h)

Автор: SVV Webhook Bot Team
Версія: 1.0.0
"""

import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Project imports
from bot import bot_instance
from settings_manager import settings
from models import db_manager, Base
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, func

# Indicators
from indicators import (
    calculate_rsi_series,
    calculate_atr_series,
    calculate_ema,
    calculate_hma
)
from rsi_screener import calculate_mfi_series, get_auto_htf

logger = logging.getLogger("SignalFlipScalper")


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class TradeStatus(Enum):
    PENDING = "Pending"
    OPEN = "Open"
    TP1_HIT = "TP1 Hit"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


# ============================================================================
#                         TIMEFRAME PRESETS (BYBIT OPTIMIZED)
# ============================================================================

TIMEFRAME_PRESETS = {
    "1": {
        "name": "Ultra Scalping (1m)",
        "tp1_percent": 0.15,
        "tp2_percent": 0.30,
        "sl_percent": 0.20,
        "trailing_offset": 0.10,
        "max_hold_minutes": 5,
        "atr_min": 0.1,
        "atr_max": 2.0,
        "min_volume_mult": 2.5,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
    },
    "5": {
        "name": "Fast Scalping (5m)",
        "tp1_percent": 0.25,
        "tp2_percent": 0.50,
        "sl_percent": 0.30,
        "trailing_offset": 0.15,
        "max_hold_minutes": 15,
        "atr_min": 0.15,
        "atr_max": 2.5,
        "min_volume_mult": 2.0,
        "rsi_oversold": 38,
        "rsi_overbought": 62,
    },
    "15": {
        "name": "Standard Scalping (15m)",
        "tp1_percent": 0.40,
        "tp2_percent": 0.80,
        "sl_percent": 0.50,
        "trailing_offset": 0.20,
        "max_hold_minutes": 45,
        "atr_min": 0.2,
        "atr_max": 3.0,
        "min_volume_mult": 1.5,
        "rsi_oversold": 40,
        "rsi_overbought": 60,
    },
    "30": {
        "name": "Slow Scalping (30m)",
        "tp1_percent": 0.60,
        "tp2_percent": 1.20,
        "sl_percent": 0.70,
        "trailing_offset": 0.30,
        "max_hold_minutes": 90,
        "atr_min": 0.3,
        "atr_max": 4.0,
        "min_volume_mult": 1.3,
        "rsi_oversold": 42,
        "rsi_overbought": 58,
    },
    "60": {
        "name": "Swing Scalping (1h)",
        "tp1_percent": 1.00,
        "tp2_percent": 2.00,
        "sl_percent": 1.00,
        "trailing_offset": 0.50,
        "max_hold_minutes": 240,
        "atr_min": 0.4,
        "atr_max": 5.0,
        "min_volume_mult": 1.2,
        "rsi_oversold": 45,
        "rsi_overbought": 55,
    },
}

# HTF Mapping
HTF_MAPPING = {
    "1": "15",
    "5": "60",
    "15": "60",
    "30": "240",
    "60": "240",
}

# Parameter descriptions
PARAM_HELP = {
    "sfs_enabled": "Увімкнути/вимкнути модуль Signal Flip Scalper",
    "sfs_timeframe": "Основний таймфрейм для аналізу",
    "sfs_auto_preset": "Автоматично застосовувати пресети при зміні TF",
    
    # Filters
    "sfs_use_rsi_filter": "Фільтр RSI Zone - вхід тільки з екстремальних зон",
    "sfs_use_mfi_filter": "Фільтр MFI - підтвердження грошового потоку",
    "sfs_use_volume_filter": "Фільтр Volume - вимагає підвищеного об'єму",
    "sfs_use_btc_filter": "Фільтр BTC Trend - не проти глобального тренду",
    "sfs_use_htf_filter": "Фільтр HTF - підтвердження старшого таймфрейму",
    "sfs_use_atr_filter": "Фільтр ATR - контроль волатильності",
    "sfs_use_time_filter": "Фільтр Time - торгівля тільки в активні години",
    "sfs_use_momentum_filter": "Фільтр Momentum - підтвердження імпульсу",
    
    # Trading
    "sfs_tp1_percent": "Take Profit 1 (%) - закриття 50% позиції",
    "sfs_tp2_percent": "Take Profit 2 (%) - закриття решти",
    "sfs_sl_percent": "Stop Loss (%)",
    "sfs_use_trailing": "Trailing Stop після TP1",
    "sfs_trailing_offset": "Відступ trailing stop (%)",
    "sfs_max_hold_minutes": "Максимальний час утримання (хвилини)",
    
    # Risk
    "sfs_max_daily_trades": "Максимум угод на день",
    "sfs_max_open_positions": "Максимум відкритих позицій",
    "sfs_position_size_percent": "Розмір позиції (% від балансу)",
    "sfs_max_daily_loss": "Максимальний денний збиток (%)",
    "sfs_leverage": "Плече",
    
    # Execution
    "sfs_paper_trading": "Paper Trading режим (без реальних угод)",
    "sfs_auto_execute": "Автоматичне виконання угод",
    "sfs_telegram_signals": "Надсилати сигнали в Telegram",
    
    # Scan
    "sfs_min_volume_24h": "Мінімальний 24h об'єм ($)",
    "sfs_scan_limit": "Кількість монет для сканування",
    "sfs_scan_interval": "Інтервал автосканування (сек)",
}


# ============================================================================
#                              DATABASE MODEL
# ============================================================================

class SignalFlipTrade(Base):
    """Модель угоди Signal Flip Scalper"""
    __tablename__ = 'signal_flip_trades'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # LONG/SHORT
    
    # Signal info
    prev_signal = Column(String(10))  # Previous signal before flip
    new_signal = Column(String(10))   # New signal after flip
    flip_reason = Column(Text)        # Why signal flipped
    
    # Prices
    signal_price = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    current_price = Column(Float)
    highest_price = Column(Float)
    lowest_price = Column(Float)
    
    # TP tracking
    tp1_hit = Column(Boolean, default=False)
    tp1_exit_price = Column(Float)
    tp1_pnl = Column(Float)
    
    # P&L
    pnl_percent = Column(Float, default=0)
    pnl_usdt = Column(Float, default=0)
    
    # Status
    status = Column(String(20), default='Pending')  # Pending/Open/TP1 Hit/Closed/Cancelled
    exit_reason = Column(String(50))
    
    # Filters passed
    filters_passed = Column(Text)  # JSON: which filters were active and passed
    
    # Indicators at entry
    rsi_at_entry = Column(Float)
    mfi_at_entry = Column(Float)
    volume_ratio = Column(Float)
    atr_percent = Column(Float)
    btc_trend = Column(String(20))
    htf_signal = Column(String(10))
    
    # Trade params
    paper_trade = Column(Boolean, default=True)
    qty = Column(Float)
    leverage = Column(Integer, default=10)
    timeframe = Column(String(10))
    
    # Timestamps
    signal_time = Column(DateTime, default=datetime.utcnow)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    hold_time_minutes = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
#                         SIGNAL TRACKER (State Machine)
# ============================================================================

class SignalTracker:
    """
    Відстежує попередні сигнали для кожного символу.
    Зберігає історію для виявлення Signal Flip.
    """
    
    def __init__(self):
        # {symbol: {'prev_signal': 'BUY', 'current_signal': 'SELL', 'flip_time': datetime}}
        self._signals: Dict[str, Dict] = defaultdict(lambda: {
            'prev_signal': SignalDirection.NONE.value,
            'current_signal': SignalDirection.NONE.value,
            'last_update': None,
            'flip_detected': False
        })
        self._lock = threading.Lock()
    
    def update_signal(self, symbol: str, new_signal: str) -> Optional[Dict]:
        """
        Оновлює сигнал для символу.
        
        Returns:
            Dict з інфо про flip якщо відбувся, інакше None
        """
        with self._lock:
            state = self._signals[symbol]
            old_signal = state['current_signal']
            
            # Оновлюємо стан
            state['prev_signal'] = old_signal
            state['current_signal'] = new_signal
            state['last_update'] = datetime.utcnow()
            
            # Перевіряємо чи був flip
            if self._is_valid_flip(old_signal, new_signal):
                state['flip_detected'] = True
                logger.info(f"🔄 Signal Flip detected: {symbol} {old_signal} → {new_signal}")
                return {
                    'symbol': symbol,
                    'prev_signal': old_signal,
                    'new_signal': new_signal,
                    'direction': 'LONG' if new_signal == 'BUY' else 'SHORT',
                    'flip_time': datetime.utcnow()
                }
            
            state['flip_detected'] = False
            return None
    
    def _is_valid_flip(self, old: str, new: str) -> bool:
        """Перевіряє чи це валідний flip (BUY↔SELL)"""
        if old == SignalDirection.NONE.value:
            return False
        if new == SignalDirection.NONE.value:
            return False
        return old != new
    
    def get_state(self, symbol: str) -> Dict:
        """Повертає поточний стан сигналу"""
        with self._lock:
            return dict(self._signals[symbol])
    
    def clear(self, symbol: str = None):
        """Очищає стан для символу або всіх"""
        with self._lock:
            if symbol:
                self._signals.pop(symbol, None)
            else:
                self._signals.clear()


# ============================================================================
#                              FILTER ENGINE
# ============================================================================

class FilterEngine:
    """
    Система фільтрів для Signal Flip Scalper.
    Всі фільтри опціональні - працює при вимкненні будь-якого.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def update_config(self, config: Dict):
        self.config = config
    
    def apply_filters(self, data: Dict, direction: str) -> Tuple[bool, List[str], Dict[str, bool]]:
        """
        Застосовує всі активні фільтри.
        
        Args:
            data: Dict з індикаторами та даними
            direction: 'LONG' або 'SHORT'
            
        Returns:
            (passed: bool, reasons: List[str], filter_results: Dict[str, bool])
        """
        reasons = []
        filter_results = {}
        
        # 1. RSI Zone Filter
        if self.config.get('sfs_use_rsi_filter', True):
            passed, reason = self._check_rsi_zone(data, direction)
            filter_results['rsi_zone'] = passed
            if not passed:
                reasons.append(reason)
        
        # 2. MFI Filter
        if self.config.get('sfs_use_mfi_filter', True):
            passed, reason = self._check_mfi(data, direction)
            filter_results['mfi'] = passed
            if not passed:
                reasons.append(reason)
        
        # 3. Volume Filter
        if self.config.get('sfs_use_volume_filter', True):
            passed, reason = self._check_volume(data)
            filter_results['volume'] = passed
            if not passed:
                reasons.append(reason)
        
        # 4. BTC Trend Filter
        if self.config.get('sfs_use_btc_filter', True):
            passed, reason = self._check_btc_trend(data, direction)
            filter_results['btc_trend'] = passed
            if not passed:
                reasons.append(reason)
        
        # 5. HTF Filter
        if self.config.get('sfs_use_htf_filter', True):
            passed, reason = self._check_htf(data, direction)
            filter_results['htf'] = passed
            if not passed:
                reasons.append(reason)
        
        # 6. ATR Filter
        if self.config.get('sfs_use_atr_filter', True):
            passed, reason = self._check_atr(data)
            filter_results['atr'] = passed
            if not passed:
                reasons.append(reason)
        
        # 7. Time Filter
        if self.config.get('sfs_use_time_filter', True):
            passed, reason = self._check_time()
            filter_results['time'] = passed
            if not passed:
                reasons.append(reason)
        
        # 8. Momentum Filter
        if self.config.get('sfs_use_momentum_filter', True):
            passed, reason = self._check_momentum(data, direction)
            filter_results['momentum'] = passed
            if not passed:
                reasons.append(reason)
        
        # Результат: passed якщо немає причин відмови
        all_passed = len(reasons) == 0
        
        return all_passed, reasons, filter_results
    
    def _check_rsi_zone(self, data: Dict, direction: str) -> Tuple[bool, str]:
        """RSI має бути в екстремальній зоні"""
        rsi = data.get('rsi', 50)
        preset = TIMEFRAME_PRESETS.get(self.config.get('sfs_timeframe', '15'), {})
        
        oversold = preset.get('rsi_oversold', 40)
        overbought = preset.get('rsi_overbought', 60)
        
        if direction == 'LONG':
            if rsi > oversold:
                return False, f"RSI {rsi:.1f} > {oversold} (not oversold)"
        else:  # SHORT
            if rsi < overbought:
                return False, f"RSI {rsi:.1f} < {overbought} (not overbought)"
        
        return True, ""
    
    def _check_mfi(self, data: Dict, direction: str) -> Tuple[bool, str]:
        """MFI підтверджує напрямок"""
        mfi = data.get('mfi', 50)
        
        if direction == 'LONG':
            # MFI < 50 або зростає
            if mfi > 55:
                return False, f"MFI {mfi:.1f} > 55 (no buy pressure)"
        else:  # SHORT
            # MFI > 50 або падає
            if mfi < 45:
                return False, f"MFI {mfi:.1f} < 45 (no sell pressure)"
        
        return True, ""
    
    def _check_volume(self, data: Dict) -> Tuple[bool, str]:
        """Volume має бути вище середнього"""
        volume_ratio = data.get('volume_ratio', 1.0)
        preset = TIMEFRAME_PRESETS.get(self.config.get('sfs_timeframe', '15'), {})
        min_mult = preset.get('min_volume_mult', 1.5)
        
        if volume_ratio < min_mult:
            return False, f"Volume ratio {volume_ratio:.2f}x < {min_mult}x"
        
        return True, ""
    
    def _check_btc_trend(self, data: Dict, direction: str) -> Tuple[bool, str]:
        """Не торгуємо проти BTC тренду"""
        btc_trend = data.get('btc_trend', 'NEUTRAL')
        
        if direction == 'LONG' and btc_trend == 'BEARISH':
            return False, f"LONG against BTC BEARISH trend"
        if direction == 'SHORT' and btc_trend == 'BULLISH':
            return False, f"SHORT against BTC BULLISH trend"
        
        return True, ""
    
    def _check_htf(self, data: Dict, direction: str) -> Tuple[bool, str]:
        """HTF сигнал підтверджує напрямок"""
        htf_signal = data.get('htf_signal', 'NONE')
        
        if direction == 'LONG' and htf_signal == 'SELL':
            return False, f"HTF signal is SELL (against LONG)"
        if direction == 'SHORT' and htf_signal == 'BUY':
            return False, f"HTF signal is BUY (against SHORT)"
        
        return True, ""
    
    def _check_atr(self, data: Dict) -> Tuple[bool, str]:
        """ATR в допустимому діапазоні"""
        atr_percent = data.get('atr_percent', 1.0)
        preset = TIMEFRAME_PRESETS.get(self.config.get('sfs_timeframe', '15'), {})
        
        atr_min = preset.get('atr_min', 0.2)
        atr_max = preset.get('atr_max', 3.0)
        
        if atr_percent < atr_min:
            return False, f"ATR {atr_percent:.2f}% < {atr_min}% (too low volatility)"
        if atr_percent > atr_max:
            return False, f"ATR {atr_percent:.2f}% > {atr_max}% (too high volatility)"
        
        return True, ""
    
    def _check_time(self) -> Tuple[bool, str]:
        """Торгівля тільки в активні години (08:00-20:00 UTC)"""
        hour = datetime.utcnow().hour
        
        if hour < 8 or hour >= 20:
            return False, f"Outside trading hours (current: {hour}:00 UTC)"
        
        return True, ""
    
    def _check_momentum(self, data: Dict, direction: str) -> Tuple[bool, str]:
        """Momentum (HMA slope) підтверджує напрямок"""
        momentum = data.get('momentum', 'neutral')
        
        if direction == 'LONG' and momentum == 'bearish':
            return False, f"Momentum is BEARISH (against LONG)"
        if direction == 'SHORT' and momentum == 'bullish':
            return False, f"Momentum is BULLISH (against SHORT)"
        
        return True, ""


# ============================================================================
#                           TRADE EXECUTOR
# ============================================================================

class TradeExecutor:
    """Виконавець угод для Bybit"""
    
    def __init__(self):
        self.session = None
    
    def _get_session(self):
        """Отримує Bybit session"""
        if self.session is None:
            from bot import bot_instance
            self.session = bot_instance.session
        return self.session
    
    def open_position(self, symbol: str, direction: str, qty: float, 
                      leverage: int, sl_price: float, tp1_price: float) -> Dict:
        """
        Відкриває позицію на Bybit.
        
        Returns:
            Dict з результатом
        """
        try:
            session = self._get_session()
            
            # Встановлюємо плече
            try:
                session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
            except Exception as e:
                logger.debug(f"Leverage already set or error: {e}")
            
            # Визначаємо сторону
            side = "Buy" if direction == "LONG" else "Sell"
            
            # Відкриваємо позицію
            order = session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(round(sl_price, self._get_price_precision(symbol))),
                takeProfit=str(round(tp1_price, self._get_price_precision(symbol))),
                tpslMode="Partial",
                tpOrderType="Market",
                slOrderType="Market"
            )
            
            if order.get('retCode') == 0:
                logger.info(f"✅ Position opened: {symbol} {direction} qty={qty}")
                return {
                    'status': 'ok',
                    'order_id': order.get('result', {}).get('orderId'),
                    'symbol': symbol,
                    'direction': direction,
                    'qty': qty
                }
            else:
                error = order.get('retMsg', 'Unknown error')
                logger.error(f"❌ Order failed: {error}")
                return {'status': 'error', 'error': error}
                
        except Exception as e:
            logger.error(f"❌ Open position error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_position(self, symbol: str, direction: str, qty: float = None) -> Dict:
        """Закриває позицію (повністю або частково)"""
        try:
            session = self._get_session()
            
            side = "Sell" if direction == "LONG" else "Buy"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "reduceOnly": True
            }
            
            if qty:
                params["qty"] = str(qty)
            else:
                # Закриваємо все
                position = self._get_position(symbol)
                if position:
                    params["qty"] = str(abs(float(position.get('size', 0))))
            
            order = session.place_order(**params)
            
            if order.get('retCode') == 0:
                logger.info(f"✅ Position closed: {symbol}")
                return {'status': 'ok'}
            else:
                return {'status': 'error', 'error': order.get('retMsg')}
                
        except Exception as e:
            logger.error(f"❌ Close position error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_sl(self, symbol: str, new_sl: float) -> Dict:
        """Оновлює Stop Loss"""
        try:
            session = self._get_session()
            
            result = session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(round(new_sl, self._get_price_precision(symbol))),
                slTriggerBy="LastPrice"
            )
            
            if result.get('retCode') == 0:
                return {'status': 'ok'}
            return {'status': 'error', 'error': result.get('retMsg')}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def calculate_qty(self, symbol: str, price: float, position_size_usdt: float) -> float:
        """Розраховує кількість з урахуванням правил Bybit"""
        try:
            session = self._get_session()
            
            info = session.get_instruments_info(category="linear", symbol=symbol)
            if info.get('retCode') != 0:
                return 0
            
            lot_filter = info['result']['list'][0].get('lotSizeFilter', {})
            qty_step = float(lot_filter.get('qtyStep', 0.001))
            min_qty = float(lot_filter.get('minOrderQty', 0.001))
            
            raw_qty = position_size_usdt / price
            qty = max(min_qty, round(raw_qty / qty_step) * qty_step)
            
            return qty
            
        except Exception as e:
            logger.error(f"Calculate qty error: {e}")
            return 0
    
    def _get_position(self, symbol: str) -> Optional[Dict]:
        """Отримує поточну позицію"""
        try:
            session = self._get_session()
            result = session.get_positions(category="linear", symbol=symbol)
            
            if result.get('retCode') == 0:
                positions = result.get('result', {}).get('list', [])
                for pos in positions:
                    if float(pos.get('size', 0)) != 0:
                        return pos
            return None
        except:
            return None
    
    def _get_price_precision(self, symbol: str) -> int:
        """Отримує точність ціни для символу"""
        try:
            session = self._get_session()
            info = session.get_instruments_info(category="linear", symbol=symbol)
            tick_size = float(info['result']['list'][0]['priceFilter']['tickSize'])
            return max(0, -int(np.log10(tick_size)))
        except:
            return 4


# ============================================================================
#                          POSITION MONITOR
# ============================================================================

class PositionMonitor:
    """Моніторинг відкритих позицій"""
    
    def __init__(self, scalper: 'SignalFlipScalper'):
        self.scalper = scalper
        self.executor = TradeExecutor()
        self._running = False
        self._thread = None
    
    def start(self):
        """Запускає моніторинг"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("📊 Position Monitor started")
    
    def stop(self):
        """Зупиняє моніторинг"""
        self._running = False
    
    def _monitor_loop(self):
        """Основний цикл моніторингу"""
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(5)
    
    def _check_positions(self):
        """Перевіряє всі відкриті позиції"""
        session = db_manager.get_session()
        
        try:
            open_trades = session.query(SignalFlipTrade).filter(
                SignalFlipTrade.status.in_(['Open', 'TP1 Hit'])
            ).all()
            
            for trade in open_trades:
                self._process_trade(trade, session)
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Check positions error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _process_trade(self, trade: SignalFlipTrade, session):
        """Обробляє одну угоду"""
        try:
            # Отримуємо поточну ціну
            ticker = bot_instance.session.get_tickers(
                category="linear", symbol=trade.symbol
            )
            if ticker.get('retCode') != 0:
                return
            
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            trade.current_price = current_price
            
            # Оновлюємо highest/lowest
            if trade.direction == 'LONG':
                trade.highest_price = max(trade.highest_price or current_price, current_price)
            else:
                trade.lowest_price = min(trade.lowest_price or current_price, current_price)
            
            config = self.scalper._load_config()
            
            # Перевірка TP1
            if not trade.tp1_hit:
                self._check_tp1(trade, current_price, config)
            
            # Перевірка TP2
            if trade.tp1_hit:
                self._check_tp2(trade, current_price, config)
                
                # Trailing Stop
                if config.get('sfs_use_trailing', True):
                    self._update_trailing(trade, current_price, config)
            
            # Перевірка SL
            self._check_sl(trade, current_price)
            
            # Перевірка Max Hold Time
            self._check_max_hold(trade, config)
            
        except Exception as e:
            logger.error(f"Process trade error {trade.symbol}: {e}")
    
    def _check_tp1(self, trade: SignalFlipTrade, price: float, config: Dict):
        """Перевіряє досягнення TP1"""
        hit = False
        if trade.direction == 'LONG' and price >= trade.tp1_price:
            hit = True
        elif trade.direction == 'SHORT' and price <= trade.tp1_price:
            hit = True
        
        if hit:
            trade.tp1_hit = True
            trade.tp1_exit_price = price
            trade.tp1_pnl = self._calc_pnl(trade.entry_price, price, trade.direction)
            trade.status = 'TP1 Hit'
            
            # Закриваємо 50%
            if not trade.paper_trade and config.get('sfs_auto_execute', False):
                self.executor.close_position(trade.symbol, trade.direction, trade.qty / 2)
            
            # Переміщуємо SL на BE
            trade.sl_price = trade.entry_price
            if not trade.paper_trade:
                self.executor.update_sl(trade.symbol, trade.entry_price)
            
            logger.info(f"🎯 TP1 Hit: {trade.symbol} +{trade.tp1_pnl:.2f}%")
    
    def _check_tp2(self, trade: SignalFlipTrade, price: float, config: Dict):
        """Перевіряє досягнення TP2"""
        hit = False
        if trade.direction == 'LONG' and price >= trade.tp2_price:
            hit = True
        elif trade.direction == 'SHORT' and price <= trade.tp2_price:
            hit = True
        
        if hit:
            self._close_trade(trade, price, 'TP2')
    
    def _check_sl(self, trade: SignalFlipTrade, price: float):
        """Перевіряє досягнення SL"""
        hit = False
        if trade.direction == 'LONG' and price <= trade.sl_price:
            hit = True
        elif trade.direction == 'SHORT' and price >= trade.sl_price:
            hit = True
        
        if hit:
            self._close_trade(trade, price, 'Stop Loss')
    
    def _check_max_hold(self, trade: SignalFlipTrade, config: Dict):
        """Перевіряє максимальний час утримання"""
        if not trade.entry_time:
            return
        
        max_hold = config.get('sfs_max_hold_minutes', 45)
        hold_time = (datetime.utcnow() - trade.entry_time).total_seconds() / 60
        
        if hold_time >= max_hold:
            self._close_trade(trade, trade.current_price, 'Max Hold Time')
    
    def _update_trailing(self, trade: SignalFlipTrade, price: float, config: Dict):
        """Оновлює Trailing Stop"""
        offset = config.get('sfs_trailing_offset', 0.2) / 100
        
        if trade.direction == 'LONG':
            new_sl = trade.highest_price * (1 - offset)
            if new_sl > trade.sl_price:
                trade.sl_price = new_sl
                if not trade.paper_trade:
                    self.executor.update_sl(trade.symbol, new_sl)
        else:
            new_sl = trade.lowest_price * (1 + offset)
            if new_sl < trade.sl_price:
                trade.sl_price = new_sl
                if not trade.paper_trade:
                    self.executor.update_sl(trade.symbol, new_sl)
    
    def _close_trade(self, trade: SignalFlipTrade, price: float, reason: str):
        """Закриває угоду"""
        trade.exit_price = price
        trade.exit_time = datetime.utcnow()
        trade.status = 'Closed'
        trade.exit_reason = reason
        trade.pnl_percent = self._calc_pnl(trade.entry_price, price, trade.direction)
        
        if trade.entry_time:
            trade.hold_time_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)
        
        # Закриваємо реальну позицію
        if not trade.paper_trade:
            self.executor.close_position(trade.symbol, trade.direction)
        
        logger.info(f"📊 Trade closed: {trade.symbol} {reason} P&L: {trade.pnl_percent:+.2f}%")
    
    def _calc_pnl(self, entry: float, exit: float, direction: str) -> float:
        """Розраховує P&L в %"""
        if direction == 'LONG':
            return ((exit - entry) / entry) * 100
        else:
            return ((entry - exit) / entry) * 100


# ============================================================================
#                       MAIN SCALPER CLASS
# ============================================================================

class SignalFlipScalper:
    """
    🔄 Signal Flip Scalper - головний клас стратегії
    """
    
    def __init__(self):
        # State
        self.is_scanning = False
        self.progress = 0
        self.status = "Idle"
        self.last_scan_time = None
        self.scan_results = []
        self.btc_trend = "NEUTRAL"
        self.today_trades = 0
        self.auto_running = False
        
        # Components
        self.signal_tracker = SignalTracker()
        self.filter_engine = FilterEngine(self._load_config())
        self.executor = TradeExecutor()
        self.position_monitor = PositionMonitor(self)
        
        # Threading
        self._stop_scan = threading.Event()
        self._auto_thread = None
        
        # Ensure DB table
        self._ensure_table()
        
        # Start position monitor
        self.position_monitor.start()
        
        logger.info("🔄 Signal Flip Scalper initialized")
    
    def _ensure_table(self):
        """Створює таблицю якщо не існує"""
        try:
            SignalFlipTrade.__table__.create(db_manager.engine, checkfirst=True)
        except Exception as e:
            logger.debug(f"Table exists or error: {e}")
    
    def _load_config(self) -> Dict:
        """Завантажує конфігурацію"""
        
        def to_bool(val, default=True):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val) if val is not None else default
        
        def to_int(val, default=0):
            try:
                return int(float(val)) if val is not None else default
            except:
                return default
        
        def to_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except:
                return default
        
        tf = str(settings.get('sfs_timeframe', '15'))
        preset = TIMEFRAME_PRESETS.get(tf, TIMEFRAME_PRESETS['15'])
        use_preset = to_bool(settings.get('sfs_auto_preset'), True)
        
        return {
            # General
            'sfs_enabled': to_bool(settings.get('sfs_enabled'), True),
            'sfs_timeframe': tf,
            'sfs_auto_preset': use_preset,
            
            # Filters (all optional)
            'sfs_use_rsi_filter': to_bool(settings.get('sfs_use_rsi_filter'), True),
            'sfs_use_mfi_filter': to_bool(settings.get('sfs_use_mfi_filter'), True),
            'sfs_use_volume_filter': to_bool(settings.get('sfs_use_volume_filter'), True),
            'sfs_use_btc_filter': to_bool(settings.get('sfs_use_btc_filter'), True),
            'sfs_use_htf_filter': to_bool(settings.get('sfs_use_htf_filter'), True),
            'sfs_use_atr_filter': to_bool(settings.get('sfs_use_atr_filter'), True),
            'sfs_use_time_filter': to_bool(settings.get('sfs_use_time_filter'), False),
            'sfs_use_momentum_filter': to_bool(settings.get('sfs_use_momentum_filter'), True),
            
            # Take Profit / Stop Loss
            'sfs_tp1_percent': to_float(settings.get('sfs_tp1_percent'), preset['tp1_percent']) if use_preset else to_float(settings.get('sfs_tp1_percent'), 0.4),
            'sfs_tp2_percent': to_float(settings.get('sfs_tp2_percent'), preset['tp2_percent']) if use_preset else to_float(settings.get('sfs_tp2_percent'), 0.8),
            'sfs_sl_percent': to_float(settings.get('sfs_sl_percent'), preset['sl_percent']) if use_preset else to_float(settings.get('sfs_sl_percent'), 0.5),
            'sfs_use_trailing': to_bool(settings.get('sfs_use_trailing'), True),
            'sfs_trailing_offset': to_float(settings.get('sfs_trailing_offset'), preset['trailing_offset']) if use_preset else to_float(settings.get('sfs_trailing_offset'), 0.2),
            'sfs_max_hold_minutes': to_int(settings.get('sfs_max_hold_minutes'), preset['max_hold_minutes']) if use_preset else to_int(settings.get('sfs_max_hold_minutes'), 45),
            
            # Risk Management
            'sfs_max_daily_trades': to_int(settings.get('sfs_max_daily_trades'), 5),
            'sfs_max_open_positions': to_int(settings.get('sfs_max_open_positions'), 2),
            'sfs_position_size_percent': to_float(settings.get('sfs_position_size_percent'), 5.0),
            'sfs_max_daily_loss': to_float(settings.get('sfs_max_daily_loss'), 3.0),
            'sfs_leverage': to_int(settings.get('sfs_leverage'), 10),
            
            # Execution
            'sfs_paper_trading': to_bool(settings.get('sfs_paper_trading'), True),
            'sfs_auto_execute': to_bool(settings.get('sfs_auto_execute'), False),
            'sfs_telegram_signals': to_bool(settings.get('sfs_telegram_signals'), False),
            
            # Scan
            'sfs_min_volume_24h': to_float(settings.get('sfs_min_volume_24h'), 5_000_000),
            'sfs_scan_limit': to_int(settings.get('sfs_scan_limit'), 50),
            'sfs_scan_interval': to_int(settings.get('sfs_scan_interval'), 60),
        }
    
    def get_config(self) -> Dict:
        """Повертає поточну конфігурацію"""
        return self._load_config()
    
    def save_config(self, data: Dict):
        """Зберігає конфігурацію"""
        settings.save_settings(data)
        self.filter_engine.update_config(self._load_config())
        logger.info("Config saved")
    
    # ========================================================================
    #                           BTC TREND ANALYSIS
    # ========================================================================
    
    def analyze_btc_trend(self) -> str:
        """Аналізує тренд BTC"""
        try:
            df = self._fetch_klines('BTCUSDT', '60', 100)
            if df is None or len(df) < 50:
                return "NEUTRAL"
            
            ema20 = calculate_ema(df['close'], 20)
            ema50 = calculate_ema(df['close'], 50)
            rsi = calculate_rsi_series(df['close'], 14)
            
            price = df['close'].iloc[-1]
            ema20_val = ema20.iloc[-1]
            ema50_val = ema50.iloc[-1]
            rsi_val = rsi.iloc[-1]
            
            bullish_signals = 0
            bearish_signals = 0
            
            if price > ema20_val:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if ema20_val > ema50_val:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if rsi_val > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if bullish_signals >= 2:
                self.btc_trend = "BULLISH"
            elif bearish_signals >= 2:
                self.btc_trend = "BEARISH"
            else:
                self.btc_trend = "NEUTRAL"
            
            return self.btc_trend
            
        except Exception as e:
            logger.error(f"BTC trend error: {e}")
            return "NEUTRAL"
    
    # ========================================================================
    #                           DATA FETCHING
    # ========================================================================
    
    def _fetch_klines(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Отримує OHLCV дані"""
        try:
            result = bot_instance.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if result.get('retCode') != 0:
                return None
            
            data = result['result']['list']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Fetch klines error {symbol}: {e}")
            return None
    
    # ========================================================================
    #                        INDICATOR CALCULATION
    # ========================================================================
    
    def _calculate_indicators(self, df: pd.DataFrame, symbol: str, config: Dict) -> Dict:
        """Розраховує всі індикатори"""
        result = {
            'rsi': 50.0,
            'mfi': 50.0,
            'volume_ratio': 1.0,
            'atr_percent': 1.0,
            'momentum': 'neutral',
            'signal': 'NONE',
            'htf_signal': 'NONE',
            'btc_trend': self.btc_trend,
            'price': 0
        }
        
        if df is None or len(df) < 50:
            return result
        
        try:
            # Price
            result['price'] = float(df['close'].iloc[-1])
            
            # RSI
            rsi_series = calculate_rsi_series(df['close'], 14)
            result['rsi'] = float(rsi_series.iloc[-1])
            
            # MFI
            mfi_series = calculate_mfi_series(
                df['high'], df['low'], df['close'], df['volume'], 20
            )
            result['mfi'] = float(mfi_series.iloc[-1])
            
            # Volume Ratio
            vol_avg = df['volume'].rolling(20).mean().iloc[-1]
            vol_current = df['volume'].iloc[-1]
            result['volume_ratio'] = vol_current / vol_avg if vol_avg > 0 else 1.0
            
            # ATR %
            atr_series = calculate_atr_series(df['high'], df['low'], df['close'], 14)
            atr = float(atr_series.iloc[-1])
            result['atr_percent'] = (atr / result['price']) * 100
            
            # Momentum (HMA based)
            hma_fast = calculate_hma(df['close'], 9)
            hma_slow = calculate_hma(df['close'], 21)
            
            if hma_fast.iloc[-1] > hma_slow.iloc[-1] and hma_fast.iloc[-1] > hma_fast.iloc[-2]:
                result['momentum'] = 'bullish'
            elif hma_fast.iloc[-1] < hma_slow.iloc[-1] and hma_fast.iloc[-1] < hma_fast.iloc[-2]:
                result['momentum'] = 'bearish'
            else:
                result['momentum'] = 'neutral'
            
            # Signal (RSI + MFI based)
            tf = config.get('sfs_timeframe', '15')
            preset = TIMEFRAME_PRESETS.get(tf, TIMEFRAME_PRESETS['15'])
            
            rsi_oversold = preset.get('rsi_oversold', 40)
            rsi_overbought = preset.get('rsi_overbought', 60)
            
            if result['rsi'] <= rsi_oversold and result['mfi'] < 45:
                result['signal'] = 'BUY'
            elif result['rsi'] >= rsi_overbought and result['mfi'] > 55:
                result['signal'] = 'SELL'
            else:
                result['signal'] = 'NONE'
            
            # HTF Signal
            htf = HTF_MAPPING.get(tf, '60')
            htf_df = self._fetch_klines(symbol, htf, 100)
            if htf_df is not None and len(htf_df) >= 50:
                htf_rsi = calculate_rsi_series(htf_df['close'], 14)
                htf_rsi_val = float(htf_rsi.iloc[-1])
                
                if htf_rsi_val <= 40:
                    result['htf_signal'] = 'BUY'
                elif htf_rsi_val >= 60:
                    result['htf_signal'] = 'SELL'
                else:
                    result['htf_signal'] = 'NONE'
            
        except Exception as e:
            logger.error(f"Calculate indicators error: {e}")
        
        return result
    
    # ========================================================================
    #                              SCANNING
    # ========================================================================
    
    def start_scan(self):
        """Запускає сканування"""
        if self.is_scanning:
            return {'status': 'error', 'error': 'Already scanning'}
        
        self._stop_scan.clear()
        thread = threading.Thread(target=self._scan_thread, daemon=True)
        thread.start()
        
        return {'status': 'started'}
    
    def stop_scan(self):
        """Зупиняє сканування"""
        self._stop_scan.set()
        self.is_scanning = False
        self.status = "Stopped"
        return {'status': 'stopped'}
    
    def _scan_thread(self):
        """Основний потік сканування"""
        self.is_scanning = True
        self.progress = 0
        self.status = "Starting..."
        self.scan_results = []
        
        config = self._load_config()
        
        try:
            # Analyze BTC
            self.status = "Analyzing BTC trend..."
            self.progress = 5
            self.analyze_btc_trend()
            
            # Get symbols
            self.status = "Fetching markets..."
            self.progress = 10
            tickers = bot_instance.get_all_tickers()
            
            min_volume = config.get('sfs_min_volume_24h', 5_000_000)
            scan_limit = config.get('sfs_scan_limit', 50)
            
            targets = [
                t for t in tickers
                if t['symbol'].endswith('USDT')
                and float(t.get('turnover24h', 0)) > min_volume
            ]
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:scan_limit]
            
            total = len(targets)
            self.status = f"Scanning {total} coins..."
            
            found_flips = []
            tf = config.get('sfs_timeframe', '15')
            
            for i, t in enumerate(targets):
                if self._stop_scan.is_set():
                    break
                
                symbol = t['symbol']
                self.status = f"Analyzing {symbol}... ({i+1}/{total})"
                self.progress = 10 + int((i / total) * 85)
                
                try:
                    # Fetch data
                    df = self._fetch_klines(symbol, tf, 200)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Calculate indicators
                    indicators = self._calculate_indicators(df, symbol, config)
                    
                    # Update signal tracker
                    flip = self.signal_tracker.update_signal(symbol, indicators['signal'])
                    
                    if flip:
                        # Apply filters
                        passed, reasons, filter_results = self.filter_engine.apply_filters(
                            indicators, flip['direction']
                        )
                        
                        flip_info = {
                            **flip,
                            'price': indicators['price'],
                            'rsi': indicators['rsi'],
                            'mfi': indicators['mfi'],
                            'volume_ratio': indicators['volume_ratio'],
                            'atr_percent': indicators['atr_percent'],
                            'btc_trend': self.btc_trend,
                            'htf_signal': indicators['htf_signal'],
                            'momentum': indicators['momentum'],
                            'filters_passed': passed,
                            'reject_reasons': reasons,
                            'filter_results': filter_results
                        }
                        
                        found_flips.append(flip_info)
                        
                        if passed:
                            logger.info(f"✅ Valid Flip: {symbol} {flip['direction']} - All filters passed")
                        else:
                            logger.debug(f"❌ Flip rejected: {symbol} - {', '.join(reasons)}")
                    
                    time.sleep(0.1)  # Rate limit
                    
                except Exception as e:
                    logger.debug(f"Scan {symbol}: {e}")
                    continue
            
            # Filter only passed signals
            valid_signals = [f for f in found_flips if f['filters_passed']]
            
            self.scan_results = valid_signals
            self.progress = 100
            self.status = f"Done! Found {len(valid_signals)} valid signals"
            self.last_scan_time = datetime.now()
            
            # Auto-execute if enabled
            if config.get('sfs_auto_execute', False) and valid_signals:
                self._auto_execute_signals(valid_signals, config)
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {e}"
        finally:
            self.is_scanning = False
    
    def _auto_execute_signals(self, signals: List[Dict], config: Dict):
        """Автоматично виконує сигнали"""
        max_trades = config.get('sfs_max_daily_trades', 5)
        max_open = config.get('sfs_max_open_positions', 2)
        
        # Count today's trades
        session = db_manager.get_session()
        today = datetime.utcnow().date()
        today_count = session.query(SignalFlipTrade).filter(
            func.date(SignalFlipTrade.created_at) == today
        ).count()
        
        # Count open positions
        open_count = session.query(SignalFlipTrade).filter(
            SignalFlipTrade.status.in_(['Open', 'TP1 Hit'])
        ).count()
        session.close()
        
        if today_count >= max_trades:
            logger.info(f"⚠️ Max daily trades reached ({max_trades})")
            return
        
        for signal in signals:
            if open_count >= max_open:
                logger.info(f"⚠️ Max open positions reached ({max_open})")
                break
            
            if today_count >= max_trades:
                break
            
            result = self.execute_signal(signal)
            if result.get('status') == 'ok':
                today_count += 1
                open_count += 1
    
    # ========================================================================
    #                          TRADE EXECUTION
    # ========================================================================
    
    def execute_signal(self, signal: Dict) -> Dict:
        """Виконує сигнал - відкриває угоду"""
        config = self._load_config()
        
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            price = signal['price']
            
            # Calculate TP/SL
            tp1_pct = config.get('sfs_tp1_percent', 0.4) / 100
            tp2_pct = config.get('sfs_tp2_percent', 0.8) / 100
            sl_pct = config.get('sfs_sl_percent', 0.5) / 100
            
            if direction == 'LONG':
                tp1_price = price * (1 + tp1_pct)
                tp2_price = price * (1 + tp2_pct)
                sl_price = price * (1 - sl_pct)
            else:
                tp1_price = price * (1 - tp1_pct)
                tp2_price = price * (1 - tp2_pct)
                sl_price = price * (1 + sl_pct)
            
            # Calculate position size
            leverage = config.get('sfs_leverage', 10)
            size_pct = config.get('sfs_position_size_percent', 5.0)
            
            balance = bot_instance.get_balance()
            position_usdt = balance * (size_pct / 100) * leverage
            qty = self.executor.calculate_qty(symbol, price, position_usdt)
            
            if qty <= 0:
                return {'status': 'error', 'error': 'Invalid quantity'}
            
            # Create trade record
            session = db_manager.get_session()
            
            trade = SignalFlipTrade(
                symbol=symbol,
                direction=direction,
                prev_signal=signal.get('prev_signal'),
                new_signal=signal.get('new_signal'),
                signal_price=price,
                entry_price=price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                current_price=price,
                highest_price=price,
                lowest_price=price,
                rsi_at_entry=signal.get('rsi'),
                mfi_at_entry=signal.get('mfi'),
                volume_ratio=signal.get('volume_ratio'),
                atr_percent=signal.get('atr_percent'),
                btc_trend=signal.get('btc_trend'),
                htf_signal=signal.get('htf_signal'),
                filters_passed=str(signal.get('filter_results', {})),
                paper_trade=config.get('sfs_paper_trading', True),
                qty=qty,
                leverage=leverage,
                timeframe=config.get('sfs_timeframe', '15'),
                status='Open',
                entry_time=datetime.utcnow()
            )
            
            session.add(trade)
            session.commit()
            trade_id = trade.id
            session.close()
            
            # Execute real trade if not paper
            if not config.get('sfs_paper_trading', True):
                result = self.executor.open_position(
                    symbol, direction, qty, leverage, sl_price, tp1_price
                )
                if result.get('status') != 'ok':
                    # Mark as cancelled
                    session = db_manager.get_session()
                    trade = session.query(SignalFlipTrade).get(trade_id)
                    trade.status = 'Cancelled'
                    trade.exit_reason = result.get('error')
                    session.commit()
                    session.close()
                    return result
            
            logger.info(f"🚀 Trade opened: {symbol} {direction} @ {price:.4f}")
            
            return {
                'status': 'ok',
                'trade_id': trade_id,
                'symbol': symbol,
                'direction': direction,
                'entry_price': price
            }
            
        except Exception as e:
            logger.error(f"Execute signal error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    #                           AUTO MODE
    # ========================================================================
    
    def start_auto_mode(self, interval: int = 60):
        """Запускає авто-режим"""
        if self.auto_running:
            return
        self.auto_running = True
        self._auto_thread = threading.Thread(
            target=self._auto_loop, args=(interval,), daemon=True
        )
        self._auto_thread.start()
        logger.info(f"🤖 Auto mode started (interval: {interval}s)")
    
    def stop_auto_mode(self):
        """Зупиняє авто-режим"""
        self.auto_running = False
        logger.info("🛑 Auto mode stopped")
    
    def _auto_loop(self, interval: int):
        """Цикл авто-режиму"""
        while self.auto_running:
            if not self.is_scanning:
                self.start_scan()
            
            # Wait for scan to complete
            while self.is_scanning and self.auto_running:
                time.sleep(1)
            
            # Wait interval
            for _ in range(interval):
                if not self.auto_running:
                    break
                time.sleep(1)
    
    # ========================================================================
    #                           STATUS & HISTORY
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Повертає поточний статус"""
        config = self._load_config()
        
        # Count today's stats
        session = db_manager.get_session()
        today = datetime.utcnow().date()
        
        today_trades = session.query(SignalFlipTrade).filter(
            func.date(SignalFlipTrade.created_at) == today
        ).count()
        
        open_positions = session.query(SignalFlipTrade).filter(
            SignalFlipTrade.status.in_(['Open', 'TP1 Hit'])
        ).count()
        
        session.close()
        
        return {
            'is_scanning': self.is_scanning,
            'progress': self.progress,
            'status': self.status,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'btc_trend': self.btc_trend,
            'auto_running': self.auto_running,
            'scan_results': len(self.scan_results),
            'today_trades': today_trades,
            'open_positions': open_positions,
            'paper_mode': config.get('sfs_paper_trading', True),
            'timeframe': config.get('sfs_timeframe', '15')
        }
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Повертає історію угод"""
        session = db_manager.get_session()
        
        try:
            trades = session.query(SignalFlipTrade)\
                .order_by(SignalFlipTrade.created_at.desc())\
                .limit(limit).all()
            
            return [{
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'prev_signal': t.prev_signal,
                'new_signal': t.new_signal,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'sl_price': t.sl_price,
                'tp1_price': t.tp1_price,
                'tp2_price': t.tp2_price,
                'tp1_hit': t.tp1_hit,
                'pnl_percent': t.pnl_percent,
                'status': t.status,
                'exit_reason': t.exit_reason,
                'rsi': t.rsi_at_entry,
                'mfi': t.mfi_at_entry,
                'volume_ratio': t.volume_ratio,
                'btc_trend': t.btc_trend,
                'paper_trade': t.paper_trade,
                'timeframe': t.timeframe,
                'hold_time': t.hold_time_minutes,
                'created_at': t.created_at.isoformat() if t.created_at else None
            } for t in trades]
            
        finally:
            session.close()
    
    def get_statistics(self) -> Dict:
        """Повертає статистику"""
        session = db_manager.get_session()
        
        try:
            trades = session.query(SignalFlipTrade).filter(
                SignalFlipTrade.status == 'Closed'
            ).all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'avg_hold_time': 0,
                    'profit_factor': 0
                }
            
            wins = [t for t in trades if t.pnl_percent > 0]
            losses = [t for t in trades if t.pnl_percent < 0]
            
            total_profit = sum(t.pnl_percent for t in wins)
            total_loss = abs(sum(t.pnl_percent for t in losses))
            
            return {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': (len(wins) / len(trades) * 100) if trades else 0,
                'total_pnl': sum(t.pnl_percent for t in trades),
                'avg_pnl': sum(t.pnl_percent for t in trades) / len(trades) if trades else 0,
                'best_trade': max(t.pnl_percent for t in trades) if trades else 0,
                'worst_trade': min(t.pnl_percent for t in trades) if trades else 0,
                'avg_hold_time': sum(t.hold_time_minutes or 0 for t in trades) / len(trades) if trades else 0,
                'profit_factor': total_profit / total_loss if total_loss > 0 else total_profit
            }
            
        finally:
            session.close()
    
    def clear_history(self):
        """Очищає історію"""
        session = db_manager.get_session()
        try:
            session.query(SignalFlipTrade).delete()
            session.commit()
            self.signal_tracker.clear()
            logger.info("History cleared")
        finally:
            session.close()


# ============================================================================
#                           GLOBAL INSTANCE
# ============================================================================

signal_flip_scalper = SignalFlipScalper()


# ============================================================================
#                           FLASK ROUTES
# ============================================================================

def register_routes(app):
    """Реєстрація маршрутів Flask"""
    from flask import render_template, request, jsonify
    
    @app.route('/signal_flip')
    def signal_flip_page():
        return render_template(
            'signal_flip_scalper.html',
            config=signal_flip_scalper.get_config(),
            status=signal_flip_scalper.get_status(),
            history=signal_flip_scalper.get_history(50),
            stats=signal_flip_scalper.get_statistics(),
            presets=TIMEFRAME_PRESETS,
            help=PARAM_HELP
        )
    
    @app.route('/signal_flip/scan', methods=['POST'])
    def signal_flip_scan():
        return jsonify(signal_flip_scalper.start_scan())
    
    @app.route('/signal_flip/stop', methods=['POST'])
    def signal_flip_stop():
        return jsonify(signal_flip_scalper.stop_scan())
    
    @app.route('/signal_flip/status')
    def signal_flip_status():
        return jsonify(signal_flip_scalper.get_status())
    
    @app.route('/signal_flip/results')
    def signal_flip_results():
        return jsonify(signal_flip_scalper.scan_results)
    
    @app.route('/signal_flip/history')
    def signal_flip_history():
        return jsonify(signal_flip_scalper.get_history())
    
    @app.route('/signal_flip/stats')
    def signal_flip_stats():
        return jsonify(signal_flip_scalper.get_statistics())
    
    @app.route('/signal_flip/config', methods=['GET', 'POST'])
    def signal_flip_config():
        if request.method == 'POST':
            data = request.json or {}
            signal_flip_scalper.save_config(data)
            return jsonify({'status': 'ok'})
        return jsonify(signal_flip_scalper.get_config())
    
    @app.route('/signal_flip/execute', methods=['POST'])
    def signal_flip_execute():
        data = request.json or {}
        return jsonify(signal_flip_scalper.execute_signal(data))
    
    @app.route('/signal_flip/auto', methods=['POST'])
    def signal_flip_auto():
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = data.get('interval', 60)
        
        if enabled:
            signal_flip_scalper.start_auto_mode(interval)
        else:
            signal_flip_scalper.stop_auto_mode()
        
        return jsonify({'status': 'ok', 'auto_running': signal_flip_scalper.auto_running})
    
    @app.route('/signal_flip/clear', methods=['POST'])
    def signal_flip_clear():
        signal_flip_scalper.clear_history()
        return jsonify({'status': 'ok'})
    
    logger.info("✅ Signal Flip Scalper routes registered")
