#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CONFLUENCE SCALPER v2.0                                     ║
║                                                                                ║
║  Професійна стратегія для швидких угод з високою ймовірністю успіху           ║
║  Об'єднує сигнали з Whale Hunter PRO, Order Block Scanner та RSI/MFI          ║
║                                                                                ║
║  NEW in v2.0:                                                                  ║
║  - Real Trade Execution via Bybit API                                         ║
║  - Position Monitor with Auto TP/SL                                           ║
║  - Analytics Engine for Self-Improvement                                       ║
║  - Problem Trade Detection & Avoidance                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, func, case

from models import db_manager, Base
from settings_manager import settings
from order_block_scanner import OrderBlockDetector, OBType

logger = logging.getLogger(__name__)


# ============================================================================
#                           ENUMS & CONSTANTS
# ============================================================================

class SignalStrength(Enum):
    WEAK = "Weak"
    MODERATE = "Moderate"
    STRONG = "Strong"
    VERY_STRONG = "Very Strong"
    EXTREME = "Extreme"


class ProblemType(Enum):
    SL_HIT_FAST = "SL Hit Fast"
    SL_HIT_REVERSAL = "SL Hit Reversal"
    EXPIRED = "Expired"
    LOW_VOLUME = "Low Volume"
    AGAINST_TREND = "Against Trend"
    HIGH_VOLATILITY = "High Volatility"
    WEAK_OB = "Weak OB"


TIMEFRAME_PRESETS = {
    "5": {
        "name": "5m Scalping",
        "description": "Ультра-швидкі угоди 5-30 хв. TP: 0.3-0.5%",
        "htf": "15", "ob_swing_length": 2, "ob_max_atr_mult": 2.0,
        "ob_distance_max": 0.8, "rsi_oversold": 35, "rsi_overbought": 65,
        "min_volume_mult": 1.3, "atr_min": 0.2, "atr_max": 3.0,
        "tp1_percent": 0.3, "tp2_percent": 0.5, "sl_atr_mult": 0.5,
        "max_hold_minutes": 30, "signal_expiry_minutes": 5, "min_confluence": 70,
    },
    "15": {
        "name": "15m Scalping",
        "description": "Швидкі угоди 15-60 хв. TP: 0.5-1%",
        "htf": "60", "ob_swing_length": 3, "ob_max_atr_mult": 2.5,
        "ob_distance_max": 1.2, "rsi_oversold": 35, "rsi_overbought": 65,
        "min_volume_mult": 1.5, "atr_min": 0.3, "atr_max": 4.0,
        "tp1_percent": 0.5, "tp2_percent": 1.0, "sl_atr_mult": 0.4,
        "max_hold_minutes": 60, "signal_expiry_minutes": 10, "min_confluence": 72,
    },
    "60": {
        "name": "1H Swing Scalp",
        "description": "Середні угоди 1-4 год. TP: 1-2%",
        "htf": "240", "ob_swing_length": 3, "ob_max_atr_mult": 3.0,
        "ob_distance_max": 2.0, "rsi_oversold": 30, "rsi_overbought": 70,
        "min_volume_mult": 1.5, "atr_min": 0.5, "atr_max": 5.0,
        "tp1_percent": 1.0, "tp2_percent": 2.0, "sl_atr_mult": 0.3,
        "max_hold_minutes": 240, "signal_expiry_minutes": 30, "min_confluence": 75,
    },
    "240": {
        "name": "4H Swing",
        "description": "Довші угоди 4-24 год. TP: 2-4%",
        "htf": "D", "ob_swing_length": 4, "ob_max_atr_mult": 3.5,
        "ob_distance_max": 3.0, "rsi_oversold": 30, "rsi_overbought": 70,
        "min_volume_mult": 1.3, "atr_min": 1.0, "atr_max": 8.0,
        "tp1_percent": 2.0, "tp2_percent": 4.0, "sl_atr_mult": 0.3,
        "max_hold_minutes": 1440, "signal_expiry_minutes": 60, "min_confluence": 78,
    }
}

PARAM_HELP = {
    "cs_enabled": "Увімкнути/вимкнути модуль Confluence Scalper.",
    "cs_timeframe": "Основний таймфрейм для аналізу.",
    "cs_auto_preset": "Автоматично застосовувати оптимальні параметри.",
    "cs_min_confluence": "Мінімальний Confluence Score (0-100) для входу.",
    "cs_weight_whale": "Вага балів Whale (RSI/MFI/RVOL) в score.",
    "cs_weight_ob": "Вага Order Block proximity в score.",
    "cs_weight_volume": "Вага об'ємного підтвердження.",
    "cs_weight_trend": "Вага трендового фільтру BTC.",
    "cs_use_btc_filter": "Торгувати тільки в напрямку BTC тренду.",
    "cs_use_volume_filter": "Перевіряти чи об'єм вище середнього.",
    "cs_use_volatility_filter": "Фільтр волатильності.",
    "cs_use_time_filter": "Торгувати тільки в активні години.",
    "cs_use_correlation_filter": "Обмежує кількість корельованих позицій.",
    "cs_ob_distance_max": "Максимальна відстань до OB у %.",
    "cs_entry_mode": "Режим входу: Retest або Immediate.",
    "cs_tp1_percent": "Take Profit 1 у %. Закривається 50%.",
    "cs_tp2_percent": "Take Profit 2 у %. Закривається решта.",
    "cs_use_trailing": "Trailing Stop після TP1.",
    "cs_trailing_offset": "Відстань trailing stop від ціни у %.",
    "cs_sl_mode": "Режим Stop Loss: OB_Edge, Fixed, ATR.",
    "cs_sl_buffer": "Додатковий буфер за OB у %.",
    "cs_max_daily_trades": "Максимум угод на день.",
    "cs_max_open_positions": "Максимум одночасно відкритих позицій.",
    "cs_position_size_percent": "Розмір позиції у % від балансу.",
    "cs_max_daily_loss": "Максимальний денний збиток у %.",
    "cs_signal_expiry": "Час життя сигналу в хвилинах.",
    "cs_max_hold_time": "Максимальний час утримання позиції.",
    "cs_scan_interval": "Інтервал сканування в секундах.",
    "cs_paper_trading": "Paper Trading - симуляція без реальних угод.",
    "cs_auto_execute": "Автоматично відкривати угоди при сигналі.",
    "cs_telegram_signals": "Відправляти сигнали в Telegram.",
    "cs_use_analytics": "Аналітика для самовдосконалення.",
    "cs_avoid_problem_symbols": "Уникати символів з високим % збитків.",
    "cs_adjust_on_losses": "Автоматично підвищувати Min Score після збитків.",
}


# ============================================================================
#                           DATABASE MODELS
# ============================================================================

class ConfluenceSignal(Base):
    __tablename__ = 'confluence_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10))
    
    confluence_score = Column(Float)
    whale_score = Column(Float)
    ob_score = Column(Float)
    volume_score = Column(Float)
    trend_score = Column(Float)
    strength = Column(String(20))
    
    signal_price = Column(Float)
    entry_price = Column(Float)
    ob_top = Column(Float)
    ob_bottom = Column(Float)
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    
    current_price = Column(Float)
    highest_price = Column(Float)
    lowest_price = Column(Float)
    
    status = Column(String(20), default='Pending')
    timeframe = Column(String(10))
    entry_mode = Column(String(20))
    btc_trend = Column(String(10))
    
    exit_price = Column(Float)
    pnl_percent = Column(Float)
    pnl_usdt = Column(Float)
    exit_reason = Column(String(50))
    
    tp1_hit = Column(Boolean, default=False)
    tp1_exit_price = Column(Float)
    tp1_pnl = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    expires_at = Column(DateTime)
    
    paper_trade = Column(Boolean, default=True)
    order_id = Column(String(50))
    qty = Column(Float)
    leverage = Column(Integer, default=10)
    
    hold_time_minutes = Column(Float)
    max_drawdown = Column(Float)
    max_profit = Column(Float)
    problem_type = Column(String(50))
    
    rsi_at_entry = Column(Float)
    mfi_at_entry = Column(Float)
    volume_ratio_at_entry = Column(Float)
    atr_at_entry = Column(Float)


class SymbolBlacklist(Base):
    __tablename__ = 'confluence_blacklist'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True)
    reason = Column(String(100))
    losses_count = Column(Integer, default=0)
    total_loss = Column(Float, default=0.0)
    last_loss_at = Column(DateTime)
    blacklisted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)


# ============================================================================
#                        DATA CLASSES
# ============================================================================

@dataclass
class ConfluenceResult:
    symbol: str
    direction: str
    confluence_score: float = 0.0
    whale_score: float = 0.0
    ob_score: float = 0.0
    volume_score: float = 0.0
    trend_score: float = 0.0
    strength: SignalStrength = SignalStrength.WEAK
    current_price: float = 0.0
    ob_top: float = 0.0
    ob_bottom: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    ob_distance_percent: float = 0.0
    atr: float = 0.0
    volume_ratio: float = 0.0
    rsi: float = 0.0
    mfi: float = 0.0
    is_valid: bool = False
    reject_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol, 'direction': self.direction,
            'confluence_score': round(self.confluence_score, 1),
            'whale_score': round(self.whale_score, 1),
            'ob_score': round(self.ob_score, 1),
            'volume_score': round(self.volume_score, 1),
            'trend_score': round(self.trend_score, 1),
            'strength': self.strength.value,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'ob_distance_percent': round(self.ob_distance_percent, 2),
            'volume_ratio': round(self.volume_ratio, 2),
            'rsi': round(self.rsi, 1),
            'mfi': round(self.mfi, 1),
            'is_valid': self.is_valid,
            'reject_reasons': self.reject_reasons
        }


# ============================================================================
#                      TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(close: pd.Series, length: int = 14) -> float:
    if len(close) < length + 1:
        return 50.0
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    first_avg_gain = gains.iloc[1:length+1].mean()
    first_avg_loss = losses.iloc[1:length+1].mean()
    avg_gain, avg_loss = first_avg_gain, first_avg_loss
    for i in range(length + 1, len(close)):
        avg_gain = (avg_gain * (length - 1) + gains.iloc[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses.iloc[i]) / length
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, length: int = 14) -> float:
    if len(close) < length + 1:
        return 50.0
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)
    positive_sum = positive_flow.rolling(length).sum().iloc[-1]
    negative_sum = negative_flow.rolling(length).sum().iloc[-1]
    if negative_sum == 0:
        return 100.0
    return 100 - (100 / (1 + positive_sum / negative_sum))


def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    if len(df) < length + 1:
        return 0.0
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(length).mean().iloc[-1])


def calculate_volume_ratio(volume: pd.Series, length: int = 20) -> float:
    if len(volume) < length:
        return 1.0
    avg_volume = volume.rolling(length).mean().iloc[-1]
    if avg_volume == 0:
        return 1.0
    return volume.iloc[-1] / avg_volume


# ============================================================================
#                        TRADE EXECUTOR
# ============================================================================

class TradeExecutor:
    def __init__(self):
        self.bot_instance = None
        self._init_bot()
    
    def _init_bot(self):
        try:
            from bot import bot_instance
            self.bot_instance = bot_instance
            logger.info("✅ Trade Executor connected")
        except Exception as e:
            logger.warning(f"Trade Executor init: {e}")
    
    def get_balance(self) -> float:
        try:
            if not self.bot_instance:
                self._init_bot()
            balance = self.bot_instance.session.get_wallet_balance(
                accountType="UNIFIED", coin="USDT"
            )
            if balance and 'result' in balance:
                for coin in balance['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        return float(coin['walletBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Get balance error: {e}")
            return 0.0
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        try:
            info = self.bot_instance.session.get_instruments_info(
                category="linear", symbol=symbol
            )
            if info and 'result' in info and info['result']['list']:
                return info['result']['list'][0]
            return None
        except:
            return None
    
    def calculate_qty(self, symbol: str, usdt_amount: float, leverage: int = 10) -> float:
        try:
            info = self.get_symbol_info(symbol)
            if not info:
                return 0.0
            ticker = self.bot_instance.session.get_tickers(category="linear", symbol=symbol)
            if not ticker or 'result' not in ticker:
                return 0.0
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            qty = (usdt_amount * leverage) / current_price
            qty_step = float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
            qty = round(qty / qty_step) * qty_step
            min_qty = float(info.get('lotSizeFilter', {}).get('minOrderQty', 0.001))
            return max(qty, min_qty)
        except Exception as e:
            logger.error(f"Calculate qty error: {e}")
            return 0.0
    
    def open_position(self, symbol: str, direction: str, qty: float,
                      sl_price: float, tp1_price: float, tp2_price: float,
                      leverage: int = 10) -> Dict:
        try:
            if not self.bot_instance:
                self._init_bot()
            side = "Buy" if direction == "LONG" else "Sell"
            try:
                self.bot_instance.session.set_leverage(
                    category="linear", symbol=symbol,
                    buyLeverage=str(leverage), sellLeverage=str(leverage)
                )
            except:
                pass
            order = self.bot_instance.session.place_order(
                category="linear", symbol=symbol, side=side,
                orderType="Market", qty=str(qty),
                stopLoss=str(round(sl_price, 6)),
                takeProfit=str(round(tp1_price, 6)),
                tpslMode="Partial", tpOrderType="Market", slOrderType="Market"
            )
            if order and order.get('retCode') == 0:
                order_id = order['result'].get('orderId')
                logger.info(f"✅ Position opened: {symbol} {direction} qty={qty}")
                return {'status': 'ok', 'order_id': order_id, 'qty': qty}
            return {'status': 'error', 'error': order.get('retMsg', 'Unknown')}
        except Exception as e:
            logger.error(f"Open position error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_position(self, symbol: str, direction: str, qty: float = None) -> Dict:
        try:
            if not self.bot_instance:
                self._init_bot()
            side = "Sell" if direction == "LONG" else "Buy"
            if qty is None:
                positions = self.bot_instance.session.get_positions(
                    category="linear", symbol=symbol
                )
                if positions and 'result' in positions:
                    for pos in positions['result']['list']:
                        if pos['symbol'] == symbol and float(pos['size']) > 0:
                            qty = float(pos['size'])
                            break
            if not qty or qty <= 0:
                return {'status': 'error', 'error': 'No position'}
            order = self.bot_instance.session.place_order(
                category="linear", symbol=symbol, side=side,
                orderType="Market", qty=str(qty), reduceOnly=True
            )
            if order and order.get('retCode') == 0:
                logger.info(f"✅ Position closed: {symbol} qty={qty}")
                return {'status': 'ok', 'qty': qty}
            return {'status': 'error', 'error': order.get('retMsg')}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def update_sl(self, symbol: str, new_sl: float) -> Dict:
        try:
            result = self.bot_instance.session.set_trading_stop(
                category="linear", symbol=symbol,
                stopLoss=str(round(new_sl, 6)), slTriggerBy="LastPrice"
            )
            if result and result.get('retCode') == 0:
                return {'status': 'ok'}
            return {'status': 'error', 'error': result.get('retMsg')}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.bot_instance.session.get_tickers(
                category="linear", symbol=symbol
            )
            if ticker and 'result' in ticker:
                return float(ticker['result']['list'][0]['lastPrice'])
            return 0.0
        except:
            return 0.0


# ============================================================================
#                        POSITION MONITOR
# ============================================================================

class PositionMonitor:
    def __init__(self, executor: TradeExecutor):
        self.executor = executor
        self.running = False
        self._thread = None
        self._stop_event = threading.Event()
    
    def start(self):
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("✅ Position Monitor started")
    
    def stop(self):
        self.running = False
        self._stop_event.set()
        logger.info("⏹️ Position Monitor stopped")
    
    def _monitor_loop(self):
        while self.running and not self._stop_event.is_set():
            try:
                self._check_positions()
                time.sleep(5)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)
    
    def _check_positions(self):
        session = db_manager.get_session()
        try:
            positions = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status.in_(['Open', 'TP1 Hit'])
            ).all()
            for pos in positions:
                self._check_single(session, pos)
            session.commit()
        except Exception as e:
            logger.error(f"Check positions error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _check_single(self, session, pos: ConfluenceSignal):
        try:
            current_price = self.executor.get_current_price(pos.symbol)
            if current_price <= 0:
                return
            
            pos.current_price = current_price
            if pos.highest_price is None or current_price > pos.highest_price:
                pos.highest_price = current_price
            if pos.lowest_price is None or current_price < pos.lowest_price:
                pos.lowest_price = current_price
            
            # P&L calculation
            if pos.direction == "LONG":
                pnl = ((current_price - pos.entry_price) / pos.entry_price) * 100
            else:
                pnl = ((pos.entry_price - current_price) / pos.entry_price) * 100
            
            # Max profit tracking
            if pos.max_profit is None or pnl > pos.max_profit:
                pos.max_profit = pnl
            
            # TP1 check
            if not pos.tp1_hit and pos.tp1_price:
                tp1_hit = (pos.direction == "LONG" and current_price >= pos.tp1_price) or \
                          (pos.direction == "SHORT" and current_price <= pos.tp1_price)
                if tp1_hit:
                    self._handle_tp1(session, pos, current_price)
            
            # TP2 check
            if pos.tp2_price:
                tp2_hit = (pos.direction == "LONG" and current_price >= pos.tp2_price) or \
                          (pos.direction == "SHORT" and current_price <= pos.tp2_price)
                if tp2_hit:
                    self._close_position(session, pos, current_price, "TP2 Hit")
                    return
            
            # SL check
            if pos.sl_price:
                sl_hit = (pos.direction == "LONG" and current_price <= pos.sl_price) or \
                         (pos.direction == "SHORT" and current_price >= pos.sl_price)
                if sl_hit:
                    pos.problem_type = self._classify_problem(pos)
                    self._close_position(session, pos, current_price, "SL Hit")
                    return
            
            # Trailing after TP1
            if pos.tp1_hit:
                self._check_trailing(pos, current_price)
            
            # Max hold time
            if pos.entry_time:
                hold_min = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
                pos.hold_time_minutes = hold_min
                config = self._get_config()
                if hold_min >= config.get('cs_max_hold_time', 60):
                    self._close_position(session, pos, current_price, "Max Hold Time")
                    
        except Exception as e:
            logger.error(f"Check {pos.symbol} error: {e}")
    
    def _handle_tp1(self, session, pos: ConfluenceSignal, price: float):
        logger.info(f"🎯 TP1 Hit: {pos.symbol} @ {price}")
        pos.tp1_hit = True
        pos.tp1_exit_price = price
        pos.status = 'TP1 Hit'
        if pos.direction == "LONG":
            pos.tp1_pnl = ((price - pos.entry_price) / pos.entry_price) * 100 * 0.5
        else:
            pos.tp1_pnl = ((pos.entry_price - price) / pos.entry_price) * 100 * 0.5
        # Close 50% and move SL to BE
        if not pos.paper_trade and pos.qty:
            close_qty = pos.qty * 0.5
            self.executor.close_position(pos.symbol, pos.direction, close_qty)
            pos.qty = pos.qty - close_qty
        if pos.entry_price:
            pos.sl_price = pos.entry_price
            if not pos.paper_trade:
                self.executor.update_sl(pos.symbol, pos.entry_price)
    
    def _check_trailing(self, pos: ConfluenceSignal, price: float):
        config = self._get_config()
        if not config.get('cs_use_trailing', True):
            return
        offset = config.get('cs_trailing_offset', 0.3)
        if pos.direction == "LONG" and pos.highest_price:
            new_sl = pos.highest_price * (1 - offset / 100)
            if new_sl > pos.sl_price:
                pos.sl_price = new_sl
                if not pos.paper_trade:
                    self.executor.update_sl(pos.symbol, new_sl)
        elif pos.direction == "SHORT" and pos.lowest_price:
            new_sl = pos.lowest_price * (1 + offset / 100)
            if new_sl < pos.sl_price:
                pos.sl_price = new_sl
                if not pos.paper_trade:
                    self.executor.update_sl(pos.symbol, new_sl)
    
    def _close_position(self, session, pos: ConfluenceSignal, price: float, reason: str):
        pos.exit_price = price
        pos.exit_time = datetime.utcnow()
        pos.exit_reason = reason
        pos.status = 'Closed'
        if pos.direction == "LONG":
            pnl = ((price - pos.entry_price) / pos.entry_price) * 100
        else:
            pnl = ((pos.entry_price - price) / pos.entry_price) * 100
        if pos.tp1_hit and pos.tp1_pnl:
            pos.pnl_percent = pos.tp1_pnl + (pnl * 0.5)
        else:
            pos.pnl_percent = pnl
        if pos.entry_time:
            pos.hold_time_minutes = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
        if not pos.paper_trade:
            self.executor.close_position(pos.symbol, pos.direction)
        logger.info(f"📊 Closed: {pos.symbol} P&L={pos.pnl_percent:.2f}% Reason={reason}")
    
    def _classify_problem(self, pos: ConfluenceSignal) -> str:
        if not pos.entry_time:
            return None
        hold_min = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
        if hold_min < 5:
            return ProblemType.SL_HIT_FAST.value
        if pos.max_profit and pos.max_profit > 0.3:
            return ProblemType.SL_HIT_REVERSAL.value
        if pos.volume_ratio_at_entry and pos.volume_ratio_at_entry < 1.0:
            return ProblemType.LOW_VOLUME.value
        if pos.atr_at_entry and pos.entry_price:
            atr_pct = (pos.atr_at_entry / pos.entry_price) * 100
            if atr_pct > 3.0:
                return ProblemType.HIGH_VOLATILITY.value
        return ProblemType.WEAK_OB.value
    
    def _get_config(self) -> Dict:
        from confluence_scalper import confluence_scalper
        return confluence_scalper.get_config()


# ============================================================================
#                        ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    def __init__(self):
        self.recommendations = []
    
    def analyze_performance(self, days: int = 7) -> Dict:
        session = db_manager.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status == 'Closed',
                ConfluenceSignal.exit_time >= cutoff
            ).all()
            
            if not trades:
                return {'message': 'No trades to analyze'}
            
            total = len(trades)
            wins = [t for t in trades if t.pnl_percent and t.pnl_percent > 0]
            losses = [t for t in trades if t.pnl_percent and t.pnl_percent <= 0]
            
            win_rate = len(wins) / total * 100 if total > 0 else 0
            total_pnl = sum(t.pnl_percent or 0 for t in trades)
            
            # Problem analysis
            problems = {}
            for t in losses:
                if t.problem_type:
                    problems[t.problem_type] = problems.get(t.problem_type, 0) + 1
            
            # Problem symbols
            symbol_stats = {}
            for t in trades:
                if t.symbol not in symbol_stats:
                    symbol_stats[t.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}
                if t.pnl_percent and t.pnl_percent > 0:
                    symbol_stats[t.symbol]['wins'] += 1
                else:
                    symbol_stats[t.symbol]['losses'] += 1
                symbol_stats[t.symbol]['pnl'] += t.pnl_percent or 0
            
            problem_symbols = [s for s, st in symbol_stats.items() 
                             if st['losses'] >= 2 and st['pnl'] < 0]
            
            # Score analysis
            score_analysis = self._analyze_by_score(trades)
            
            # Generate recommendations
            self.recommendations = self._generate_recommendations(
                win_rate, problems, problem_symbols, score_analysis
            )
            
            return {
                'period_days': days, 'total_trades': total,
                'wins': len(wins), 'losses': len(losses),
                'win_rate': round(win_rate, 1),
                'total_pnl': round(total_pnl, 2),
                'problems': problems,
                'problem_symbols': problem_symbols,
                'score_analysis': score_analysis,
                'recommendations': self.recommendations
            }
        except Exception as e:
            return {'error': str(e)}
        finally:
            session.close()
    
    def _analyze_by_score(self, trades) -> Dict:
        ranges = {
            '70-75': {'trades': 0, 'wins': 0, 'pnl': 0},
            '75-80': {'trades': 0, 'wins': 0, 'pnl': 0},
            '80-85': {'trades': 0, 'wins': 0, 'pnl': 0},
            '85-90': {'trades': 0, 'wins': 0, 'pnl': 0},
            '90+': {'trades': 0, 'wins': 0, 'pnl': 0},
        }
        for t in trades:
            score = t.confluence_score or 0
            if score >= 90: key = '90+'
            elif score >= 85: key = '85-90'
            elif score >= 80: key = '80-85'
            elif score >= 75: key = '75-80'
            else: key = '70-75'
            ranges[key]['trades'] += 1
            if t.pnl_percent and t.pnl_percent > 0:
                ranges[key]['wins'] += 1
            ranges[key]['pnl'] += t.pnl_percent or 0
        for k, v in ranges.items():
            v['win_rate'] = round(v['wins'] / v['trades'] * 100, 1) if v['trades'] > 0 else 0
        return ranges
    
    def _generate_recommendations(self, win_rate, problems, problem_symbols, score_analysis) -> List[str]:
        recs = []
        if win_rate < 60:
            recs.append(f"⚠️ Win rate низький ({win_rate:.1f}%). Підвищіть Min Score до 80+")
        
        best_range = max(score_analysis.items(), key=lambda x: x[1]['win_rate'])
        if best_range[1]['win_rate'] > 0:
            recs.append(f"📊 Найкращий score: {best_range[0]} (WR: {best_range[1]['win_rate']}%)")
        
        if problems:
            most = max(problems.items(), key=lambda x: x[1])
            if most[0] == ProblemType.SL_HIT_FAST.value:
                recs.append(f"⚡ Багато швидких SL ({most[1]}). Збільшіть SL buffer або Retest entry")
            elif most[0] == ProblemType.SL_HIT_REVERSAL.value:
                recs.append(f"🔄 SL після profit ({most[1]}). Зменшіть TP1 або активніший trailing")
        
        if problem_symbols:
            recs.append(f"🚫 Проблемні символи: {', '.join(problem_symbols[:5])}")
        
        return recs
    
    def update_blacklist(self):
        session = db_manager.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=30)
            stats = session.query(
                ConfluenceSignal.symbol,
                func.count(ConfluenceSignal.id).label('total'),
                func.sum(case((ConfluenceSignal.pnl_percent < 0, 1), else_=0)).label('losses'),
                func.sum(ConfluenceSignal.pnl_percent).label('total_pnl')
            ).filter(
                ConfluenceSignal.status == 'Closed',
                ConfluenceSignal.exit_time >= cutoff
            ).group_by(ConfluenceSignal.symbol).all()
            
            for stat in stats:
                symbol, total, losses, total_pnl = stat
                if losses and losses >= 3 and total_pnl and total_pnl < -2:
                    existing = session.query(SymbolBlacklist).filter_by(symbol=symbol).first()
                    if existing:
                        existing.losses_count = losses
                        existing.total_loss = total_pnl
                        existing.expires_at = datetime.utcnow() + timedelta(days=7)
                    else:
                        entry = SymbolBlacklist(
                            symbol=symbol,
                            reason=f"{losses} losses, {total_pnl:.2f}% PnL",
                            losses_count=losses, total_loss=total_pnl,
                            expires_at=datetime.utcnow() + timedelta(days=7)
                        )
                        session.add(entry)
                        logger.info(f"🚫 Blacklisted: {symbol}")
            
            session.query(SymbolBlacklist).filter(
                SymbolBlacklist.expires_at < datetime.utcnow()
            ).delete()
            session.commit()
        except Exception as e:
            logger.error(f"Update blacklist error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_blacklist(self) -> List[str]:
        session = db_manager.get_session()
        try:
            entries = session.query(SymbolBlacklist).filter(
                SymbolBlacklist.expires_at > datetime.utcnow()
            ).all()
            return [e.symbol for e in entries]
        finally:
            session.close()
    
    def auto_adjust_settings(self):
        session = db_manager.get_session()
        try:
            recent = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status == 'Closed'
            ).order_by(ConfluenceSignal.exit_time.desc()).limit(5).all()
            
            if len(recent) < 5:
                return
            
            losses = sum(1 for t in recent if t.pnl_percent and t.pnl_percent < 0)
            
            if losses >= 4:
                current = int(settings.get('cs_min_confluence', 72))
                new_min = min(current + 5, 90)
                settings.save_settings({'cs_min_confluence': new_min})
                logger.warning(f"⚠️ Auto-adjusted: Min Confluence {current} -> {new_min}")
            elif losses == 0:
                current = int(settings.get('cs_min_confluence', 72))
                new_min = max(current - 2, 65)
                settings.save_settings({'cs_min_confluence': new_min})
                logger.info(f"📈 Auto-adjusted: Min Confluence {current} -> {new_min}")
        except Exception as e:
            logger.error(f"Auto adjust error: {e}")
        finally:
            session.close()


# ============================================================================
#                     CONFLUENCE SCALPER ENGINE
# ============================================================================

class ConfluenceScalper:
    def __init__(self):
        self.is_scanning = False
        self.auto_running = False
        self.progress = 0
        self.status = "Idle"
        self.last_scan_time = None
        self.scan_results: List[ConfluenceResult] = []
        self.btc_trend = "NEUTRAL"
        self._stop_scan = threading.Event()
        self._auto_thread = None
        
        self.today_trades = 0
        self.today_pnl = 0.0
        self.today_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        self.executor = TradeExecutor()
        self.monitor = PositionMonitor(self.executor)
        self.analytics = AnalyticsEngine()
        
        self._ensure_tables()
        self._load_daily_stats()
        
        # 🆕 Автозапуск auto mode якщо увімкнений в налаштуваннях
        config = self.get_config()
        if config.get('cs_auto_mode', False):
            interval = config.get('cs_scan_interval', 30)
            self.start_auto_mode(interval)
            logger.info(f"✅ Confluence Scalper v2.0 initialized (auto mode: {interval}s)")
        else:
            logger.info("✅ Confluence Scalper v2.0 initialized")
    
    def _ensure_tables(self):
        try:
            ConfluenceSignal.__table__.create(db_manager.engine, checkfirst=True)
            SymbolBlacklist.__table__.create(db_manager.engine, checkfirst=True)
            
            # Migrate: add new columns to existing table
            from sqlalchemy import text
            new_columns = [
                ("current_price", "FLOAT"),
                ("highest_price", "FLOAT"),
                ("lowest_price", "FLOAT"),
                ("btc_trend", "VARCHAR(10)"),
                ("pnl_usdt", "FLOAT"),
                ("tp1_hit", "BOOLEAN DEFAULT FALSE"),
                ("tp1_exit_price", "FLOAT"),
                ("tp1_pnl", "FLOAT"),
                ("leverage", "INTEGER DEFAULT 10"),
                ("hold_time_minutes", "FLOAT"),
                ("max_drawdown", "FLOAT"),
                ("max_profit", "FLOAT"),
                ("problem_type", "VARCHAR(50)"),
                ("rsi_at_entry", "FLOAT"),
                ("mfi_at_entry", "FLOAT"),
                ("volume_ratio_at_entry", "FLOAT"),
                ("atr_at_entry", "FLOAT"),
            ]
            
            with db_manager.engine.connect() as conn:
                for col_name, col_type in new_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE confluence_signals ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        logger.info(f"✅ Added column: {col_name}")
                    except Exception:
                        pass  # Column already exists
                        
        except Exception as e:
            logger.warning(f"Table creation: {e}")
    
    def _load_daily_stats(self):
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today != self.today_date:
            self.today_trades = 0
            self.today_pnl = 0.0
            self.today_date = today
            self.analytics.auto_adjust_settings()
            self.analytics.update_blacklist()
        else:
            session = db_manager.get_session()
            try:
                today_start = datetime.strptime(today, '%Y-%m-%d')
                trades = session.query(ConfluenceSignal).filter(
                    ConfluenceSignal.created_at >= today_start
                ).all()
                self.today_trades = len([t for t in trades if t.status in ['Open', 'TP1 Hit', 'Closed']])
                self.today_pnl = sum(t.pnl_percent or 0 for t in trades if t.status == 'Closed')
            except:
                pass
            finally:
                session.close()
    
    def get_config(self) -> Dict:
        def to_bool(v, d=True):
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.lower() in ('true', '1', 'yes', 'on')
            return bool(v) if v is not None else d
        def to_int(v, d=0):
            try: return int(float(v)) if v is not None else d
            except: return d
        def to_float(v, d=0.0):
            try: return float(v) if v is not None else d
            except: return d
        
        tf = settings.get('cs_timeframe', '15')
        preset = TIMEFRAME_PRESETS.get(tf, TIMEFRAME_PRESETS['15'])
        use_preset = to_bool(settings.get('cs_auto_preset'), True)
        
        return {
            'cs_enabled': to_bool(settings.get('cs_enabled'), True),
            'cs_timeframe': tf,
            'cs_auto_preset': use_preset,
            'cs_min_confluence': to_int(settings.get('cs_min_confluence'), preset['min_confluence'] if use_preset else 50),
            'cs_weight_whale': to_int(settings.get('cs_weight_whale'), 30),
            'cs_weight_ob': to_int(settings.get('cs_weight_ob'), 30),
            'cs_weight_volume': to_int(settings.get('cs_weight_volume'), 20),
            'cs_weight_trend': to_int(settings.get('cs_weight_trend'), 20),
            'cs_use_btc_filter': to_bool(settings.get('cs_use_btc_filter'), True),
            'cs_use_volume_filter': to_bool(settings.get('cs_use_volume_filter'), True),
            'cs_use_volatility_filter': to_bool(settings.get('cs_use_volatility_filter'), True),
            'cs_use_time_filter': to_bool(settings.get('cs_use_time_filter'), False),
            'cs_use_correlation_filter': to_bool(settings.get('cs_use_correlation_filter'), True),
            'cs_ob_distance_max': to_float(settings.get('cs_ob_distance_max'), preset['ob_distance_max'] if use_preset else 1.5),
            'cs_ob_swing_length': to_int(settings.get('cs_ob_swing_length'), preset['ob_swing_length'] if use_preset else 3),
            'cs_entry_mode': settings.get('cs_entry_mode', 'Retest'),
            'cs_tp1_percent': to_float(settings.get('cs_tp1_percent'), preset['tp1_percent'] if use_preset else 0.5),
            'cs_tp2_percent': to_float(settings.get('cs_tp2_percent'), preset['tp2_percent'] if use_preset else 1.0),
            'cs_use_trailing': to_bool(settings.get('cs_use_trailing'), True),
            'cs_trailing_offset': to_float(settings.get('cs_trailing_offset'), 0.3),
            'cs_sl_mode': settings.get('cs_sl_mode', 'OB_Edge'),
            'cs_sl_atr_mult': to_float(settings.get('cs_sl_atr_mult'), preset['sl_atr_mult'] if use_preset else 0.4),
            'cs_sl_buffer': to_float(settings.get('cs_sl_buffer'), 0.1),
            'cs_max_daily_trades': to_int(settings.get('cs_max_daily_trades'), 3),
            'cs_max_open_positions': to_int(settings.get('cs_max_open_positions'), 2),
            'cs_max_same_direction': to_int(settings.get('cs_max_same_direction'), 2),
            'cs_position_size_percent': to_float(settings.get('cs_position_size_percent'), 5.0),
            'cs_max_daily_loss': to_float(settings.get('cs_max_daily_loss'), 3.0),
            'cs_leverage': to_int(settings.get('cs_leverage'), 10),
            'cs_signal_expiry': to_int(settings.get('cs_signal_expiry'), preset['signal_expiry_minutes'] if use_preset else 10),
            'cs_max_hold_time': to_int(settings.get('cs_max_hold_time'), preset['max_hold_minutes'] if use_preset else 60),
            'cs_scan_interval': to_int(settings.get('cs_scan_interval'), 30),
            'cs_atr_min': preset['atr_min'] if use_preset else 0.3,
            'cs_atr_max': preset['atr_max'] if use_preset else 4.0,
            'cs_min_volume_mult': preset['min_volume_mult'] if use_preset else 1.5,
            'cs_rsi_oversold': preset['rsi_oversold'] if use_preset else 35,
            'cs_rsi_overbought': preset['rsi_overbought'] if use_preset else 65,
            'cs_paper_trading': to_bool(settings.get('cs_paper_trading'), True),
            'cs_auto_execute': to_bool(settings.get('cs_auto_execute'), False),
            'cs_telegram_signals': to_bool(settings.get('cs_telegram_signals'), False),
            'cs_use_analytics': to_bool(settings.get('cs_use_analytics'), True),
            'cs_avoid_problem_symbols': to_bool(settings.get('cs_avoid_problem_symbols'), True),
            'cs_adjust_on_losses': to_bool(settings.get('cs_adjust_on_losses'), True),
            'cs_auto_mode': to_bool(settings.get('cs_auto_mode'), False),
        }
    
    def get_status(self) -> Dict:
        config = self.get_config()
        session = db_manager.get_session()
        try:
            open_pos = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status.in_(['Open', 'TP1 Hit'])
            ).count()
        except:
            open_pos = 0
        finally:
            session.close()
        
        return {
            'is_scanning': self.is_scanning,
            'auto_running': self.auto_running,
            'progress': self.progress,
            'status': self.status,
            'last_scan_time': self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else None,
            'btc_trend': self.btc_trend,
            'results_count': len(self.scan_results),
            'today_trades': self.today_trades,
            'today_pnl': round(self.today_pnl, 2),
            'max_daily_trades': config['cs_max_daily_trades'],
            'can_trade': self.today_trades < config['cs_max_daily_trades'],
            'paper_mode': config['cs_paper_trading'],
            'open_positions': open_pos,
            'monitor_running': self.monitor.running,
        }
    
    def analyze_btc_trend(self) -> str:
        try:
            from bot import bot_instance
            klines = bot_instance.session.get_kline(
                category="linear", symbol="BTCUSDT", interval="60", limit=50
            )
            if not klines or 'result' not in klines:
                return "NEUTRAL"
            data = klines['result']['list']
            if len(data) < 30:
                return "NEUTRAL"
            closes = [float(k[4]) for k in reversed(data)]
            ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
            current = closes[-1]
            rsi = calculate_rsi(pd.Series(closes), 14)
            if current > ema20 > ema50 and rsi > 50:
                self.btc_trend = "BULLISH"
            elif current < ema20 < ema50 and rsi < 50:
                self.btc_trend = "BEARISH"
            else:
                self.btc_trend = "NEUTRAL"
            logger.info(f"📊 BTC: {self.btc_trend} (${current:.0f}, RSI {rsi:.1f})")
            return self.btc_trend
        except Exception as e:
            logger.error(f"BTC trend error: {e}")
            return "NEUTRAL"
    
    def calculate_confluence(self, df: pd.DataFrame, symbol: str, direction: str, config: Dict) -> ConfluenceResult:
        result = ConfluenceResult(symbol=symbol, direction=direction)
        if df is None or len(df) < 50:
            result.reject_reasons.append("Insufficient data")
            return result
        
        current_price = df['close'].iloc[-1]
        result.current_price = current_price
        
        # 1. WHALE SCORE
        whale_score = 0.0
        rsi = calculate_rsi(df['close'], 14)
        result.rsi = rsi
        if direction == "LONG":
            if rsi <= config['cs_rsi_oversold']: whale_score += 40
            elif rsi < 50: whale_score += 20
        else:
            if rsi >= config['cs_rsi_overbought']: whale_score += 40
            elif rsi > 50: whale_score += 20
        
        mfi = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)
        result.mfi = mfi
        if direction == "LONG":
            if mfi <= 30: whale_score += 30
            elif mfi < 50: whale_score += 15
        else:
            if mfi >= 70: whale_score += 30
            elif mfi > 50: whale_score += 15
        
        volume_ratio = calculate_volume_ratio(df['volume'], 20)
        result.volume_ratio = volume_ratio
        if volume_ratio >= config['cs_min_volume_mult']: whale_score += 30
        elif volume_ratio >= 1.0: whale_score += 15
        result.whale_score = min(100, whale_score)
        
        # 2. ORDER BLOCK SCORE
        ob_score = 0.0
        try:
            detector = OrderBlockDetector(swing_length=config['cs_ob_swing_length'], max_atr_mult=3.0)
            bullish_obs, bearish_obs = detector.detect_order_blocks(
                df, direction.replace('LONG', 'BUY').replace('SHORT', 'SELL')
            )
            obs = bullish_obs if direction == "LONG" else bearish_obs
            valid_obs = [ob for ob in obs if ob.is_valid()]
            if valid_obs:
                nearest_ob = min(valid_obs, key=lambda ob: abs((ob.top + ob.bottom) / 2 - current_price))
                ob_mid = (nearest_ob.top + nearest_ob.bottom) / 2
                dist_pct = abs(current_price - ob_mid) / current_price * 100
                result.ob_top = nearest_ob.top
                result.ob_bottom = nearest_ob.bottom
                result.ob_distance_percent = dist_pct
                max_dist = config['cs_ob_distance_max']
                if dist_pct <= max_dist:
                    ob_score = 100 * (1 - dist_pct / max_dist)
                    atr = calculate_atr(df, 10)
                    result.atr = atr
                    if direction == "LONG":
                        result.entry_price = nearest_ob.top
                        result.sl_price = nearest_ob.bottom - (atr * config['cs_sl_atr_mult']) - (current_price * config['cs_sl_buffer'] / 100)
                        result.tp1_price = current_price * (1 + config['cs_tp1_percent'] / 100)
                        result.tp2_price = current_price * (1 + config['cs_tp2_percent'] / 100)
                    else:
                        result.entry_price = nearest_ob.bottom
                        result.sl_price = nearest_ob.top + (atr * config['cs_sl_atr_mult']) + (current_price * config['cs_sl_buffer'] / 100)
                        result.tp1_price = current_price * (1 - config['cs_tp1_percent'] / 100)
                        result.tp2_price = current_price * (1 - config['cs_tp2_percent'] / 100)
                elif dist_pct <= max_dist * 2:
                    # OB знайдено, але далекувато - часткові бали
                    ob_score = 30 * (1 - (dist_pct - max_dist) / max_dist)
                    result.reject_reasons.append(f"OB moderate: {dist_pct:.2f}%")
                else:
                    ob_score = 10  # Мінімальні бали за наявність OB
                    result.reject_reasons.append(f"OB too far: {dist_pct:.2f}%")
            else:
                # Немає OB - часткові бали якщо інші показники хороші
                ob_score = 20 if result.whale_score >= 50 else 0
                result.reject_reasons.append("No valid OB (partial score given)")
        except Exception as e:
            ob_score = 10  # Помилка детекції - мінімальні бали
            result.reject_reasons.append("OB detection issue")
        result.ob_score = ob_score
        
        # 3. VOLUME SCORE
        volume_score = 50
        if config['cs_use_volume_filter']:
            if volume_ratio >= config['cs_min_volume_mult'] * 1.5: volume_score = 100
            elif volume_ratio >= config['cs_min_volume_mult']: volume_score = 70
            elif volume_ratio >= 1.0: volume_score = 40
            else: result.reject_reasons.append(f"Low volume: {volume_ratio:.2f}x")
        result.volume_score = volume_score
        
        # 4. TREND SCORE
        trend_score = 50
        if config['cs_use_btc_filter']:
            if self.btc_trend == "BULLISH" and direction == "LONG": trend_score = 100
            elif self.btc_trend == "BEARISH" and direction == "SHORT": trend_score = 100
            elif self.btc_trend == "NEUTRAL": trend_score = 50
            else:
                trend_score = 0
                result.reject_reasons.append(f"Against BTC: {self.btc_trend}")
        result.trend_score = trend_score
        
        # FINAL SCORE
        w = {
            'whale': config['cs_weight_whale'] / 100,
            'ob': config['cs_weight_ob'] / 100,
            'volume': config['cs_weight_volume'] / 100,
            'trend': config['cs_weight_trend'] / 100,
        }
        total_w = sum(w.values())
        if total_w > 0:
            w = {k: v / total_w for k, v in w.items()}
        result.confluence_score = (
            result.whale_score * w['whale'] +
            result.ob_score * w['ob'] +
            result.volume_score * w['volume'] +
            result.trend_score * w['trend']
        )
        
        s = result.confluence_score
        if s >= 95: result.strength = SignalStrength.EXTREME
        elif s >= 85: result.strength = SignalStrength.VERY_STRONG
        elif s >= 75: result.strength = SignalStrength.STRONG
        elif s >= 65: result.strength = SignalStrength.MODERATE
        else: result.strength = SignalStrength.WEAK
        
        # Якщо немає ATR (не знайдено OB), обчислюємо його для фільтрів
        if result.atr == 0:
            result.atr = calculate_atr(df, 10)
        
        # Якщо немає SL/TP (не знайдено близький OB), встановлюємо базові
        if result.sl_price == 0 and result.atr > 0:
            atr_mult = config.get('cs_sl_atr_mult', 0.4)
            if direction == "LONG":
                result.entry_price = current_price
                result.sl_price = current_price - (result.atr * atr_mult * 2)
                result.tp1_price = current_price * (1 + config['cs_tp1_percent'] / 100)
                result.tp2_price = current_price * (1 + config['cs_tp2_percent'] / 100)
            else:
                result.entry_price = current_price
                result.sl_price = current_price + (result.atr * atr_mult * 2)
                result.tp1_price = current_price * (1 - config['cs_tp1_percent'] / 100)
                result.tp2_price = current_price * (1 - config['cs_tp2_percent'] / 100)
        
        # Additional filters
        if config['cs_use_volatility_filter'] and result.atr > 0:
            atr_pct = (result.atr / current_price) * 100
            if atr_pct < config['cs_atr_min']:
                result.reject_reasons.append(f"Low volatility: {atr_pct:.2f}%")
            elif atr_pct > config['cs_atr_max']:
                result.reject_reasons.append(f"High volatility: {atr_pct:.2f}%")
        
        if config['cs_use_time_filter']:
            hour = datetime.utcnow().hour
            if hour < 8 or hour >= 20:
                result.reject_reasons.append(f"Outside hours: {hour}:00")
        
        if config['cs_avoid_problem_symbols']:
            blacklist = self.analytics.get_blacklist()
            if symbol in blacklist:
                result.reject_reasons.append("Symbol blacklisted")
        
        # Check for critical rejections only (not informational)
        critical_rejections = [r for r in result.reject_reasons if any(
            x in r.lower() for x in ['blacklist', 'against btc', 'outside hours']
        )]
        
        result.is_valid = (
            result.confluence_score >= config['cs_min_confluence'] and
            len(critical_rejections) == 0 and
            result.ob_score >= 0  # OB не обов'язковий, якщо confluence достатній
        )
        
        # Логування для діагностики
        if not result.is_valid and result.confluence_score > 40:
            logger.debug(f"⏭️ {symbol} {direction}: score={result.confluence_score:.1f}, "
                        f"rejections={result.reject_reasons}, ob={result.ob_score:.1f}")
        return result
    
    def start_scan(self) -> bool:
        if self.is_scanning:
            return False
        config = self.get_config()
        if self.today_trades >= config['cs_max_daily_trades']:
            self.status = f"Daily limit: {self.today_trades}/{config['cs_max_daily_trades']}"
            return False
        self._stop_scan.clear()
        self.is_scanning = True
        threading.Thread(target=self._scan_thread, daemon=True).start()
        return True
    
    def stop_scan(self):
        self._stop_scan.set()
        self.is_scanning = False
    
    def _scan_thread(self):
        self.progress = 0
        self.status = "Initializing..."
        self.scan_results = []
        config = self.get_config()
        
        try:
            from bot import bot_instance
            
            self.status = "Analyzing BTC..."
            self.progress = 5
            self.analyze_btc_trend()
            
            if config['cs_use_btc_filter'] and self.btc_trend == "NEUTRAL":
                self.status = "BTC NEUTRAL - waiting"
                self.progress = 100
                self.is_scanning = False
                return
            
            self.status = "Fetching markets..."
            self.progress = 10
            tickers = bot_instance.get_all_tickers()
            
            blacklist = self.analytics.get_blacklist() if config['cs_avoid_problem_symbols'] else []
            targets = [
                t for t in tickers
                if t['symbol'].endswith('USDT')
                and float(t.get('turnover24h', 0)) > 5_000_000
                and t['symbol'] not in blacklist
            ]
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:100]
            
            total = len(targets)
            self.status = f"Scanning {total} coins..."
            
            directions = ["LONG"] if self.btc_trend == "BULLISH" else ["SHORT"] if self.btc_trend == "BEARISH" else ["LONG", "SHORT"]
            if not config['cs_use_btc_filter']:
                directions = ["LONG", "SHORT"]
            
            found = []
            rejected_count = 0
            for i, t in enumerate(targets):
                if self._stop_scan.is_set():
                    break
                symbol = t['symbol']
                self.status = f"Analyzing {symbol}... ({i+1}/{total})"
                self.progress = 10 + int((i / total) * 85)
                
                try:
                    df = self._fetch_klines(bot_instance, symbol, config['cs_timeframe'], 200)
                    if df is None or len(df) < 50:
                        continue
                    for direction in directions:
                        result = self.calculate_confluence(df, symbol, direction, config)
                        if result.is_valid:
                            found.append(result)
                            logger.info(f"✅ {symbol} {direction}: {result.confluence_score:.1f} "
                                       f"(W:{result.whale_score:.0f} OB:{result.ob_score:.0f} "
                                       f"V:{result.volume_score:.0f} T:{result.trend_score:.0f})")
                        elif result.confluence_score >= 40 and rejected_count < 10:
                            # Логуємо перші 10 близьких до порогу для діагностики
                            logger.debug(f"⏭️ {symbol} {direction}: {result.confluence_score:.1f} "
                                        f"reasons={result.reject_reasons[:2]}")
                            rejected_count += 1
                    time.sleep(0.1)  # Rate limit: 10 requests/sec max
                except Exception as e:
                    logger.debug(f"Scan {symbol}: {e}")
                    continue
            
            found.sort(key=lambda x: x.confluence_score, reverse=True)
            max_sig = max(1, (config['cs_max_daily_trades'] - self.today_trades) * 2)
            self.scan_results = found[:max_sig]
            
            self.progress = 100
            self.status = f"Done! Found {len(self.scan_results)} signals"
            self.last_scan_time = datetime.now()
            logger.info(f"📊 Scan complete: {len(self.scan_results)} signals")
            
            if self.scan_results and config['cs_auto_execute']:
                best = self.scan_results[0]
                if best.confluence_score >= config['cs_min_confluence']:
                    self._execute_signal(best, config)
                    
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
    
    def _fetch_klines(self, bot_instance, symbol: str, tf: str, limit: int):
        try:
            klines = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=tf, limit=limit
            )
            if not klines or 'result' not in klines:
                return None
            data = klines['result']['list']
            if len(data) < 50:
                return None
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 
                           'close': 'float64', 'volume': 'float64'})
            return df.iloc[::-1].reset_index(drop=True)
        except:
            return None
    
    def _execute_signal(self, signal: ConfluenceResult, config: Dict):
        session = db_manager.get_session()
        try:
            # Check limits
            open_count = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status.in_(['Open', 'TP1 Hit'])
            ).count()
            if open_count >= config['cs_max_open_positions']:
                logger.info(f"⚠️ Max positions: {open_count}")
                return
            
            if config['cs_use_correlation_filter']:
                same_dir = session.query(ConfluenceSignal).filter(
                    ConfluenceSignal.status.in_(['Open', 'TP1 Hit']),
                    ConfluenceSignal.direction == signal.direction
                ).count()
                if same_dir >= config['cs_max_same_direction']:
                    logger.info(f"⚠️ Max same direction: {same_dir}")
                    return
            
            db_signal = ConfluenceSignal(
                symbol=signal.symbol, direction=signal.direction,
                confluence_score=signal.confluence_score,
                whale_score=signal.whale_score, ob_score=signal.ob_score,
                volume_score=signal.volume_score, trend_score=signal.trend_score,
                strength=signal.strength.value,
                signal_price=signal.current_price, entry_price=signal.current_price,
                ob_top=signal.ob_top, ob_bottom=signal.ob_bottom,
                sl_price=signal.sl_price, tp1_price=signal.tp1_price, tp2_price=signal.tp2_price,
                current_price=signal.current_price,
                highest_price=signal.current_price, lowest_price=signal.current_price,
                timeframe=config['cs_timeframe'], entry_mode=config['cs_entry_mode'],
                btc_trend=self.btc_trend, paper_trade=config['cs_paper_trading'],
                leverage=config['cs_leverage'], entry_time=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=config['cs_signal_expiry']),
                rsi_at_entry=signal.rsi, mfi_at_entry=signal.mfi,
                volume_ratio_at_entry=signal.volume_ratio, atr_at_entry=signal.atr,
            )
            
            if not config['cs_paper_trading']:
                balance = self.executor.get_balance()
                pos_usdt = balance * (config['cs_position_size_percent'] / 100)
                qty = self.executor.calculate_qty(signal.symbol, pos_usdt, config['cs_leverage'])
                if qty > 0:
                    result = self.executor.open_position(
                        symbol=signal.symbol, direction=signal.direction, qty=qty,
                        sl_price=signal.sl_price, tp1_price=signal.tp1_price,
                        tp2_price=signal.tp2_price, leverage=config['cs_leverage']
                    )
                    if result['status'] == 'ok':
                        db_signal.order_id = result.get('order_id')
                        db_signal.qty = result.get('qty')
                        db_signal.status = 'Open'
                        logger.info(f"✅ Real trade: {signal.symbol} {signal.direction}")
                    else:
                        db_signal.status = 'Cancelled'
                        db_signal.exit_reason = result.get('error')
                else:
                    db_signal.status = 'Cancelled'
                    db_signal.exit_reason = 'Invalid qty'
            else:
                db_signal.status = 'Open'
                logger.info(f"📝 Paper trade: {signal.symbol} {signal.direction}")
            
            session.add(db_signal)
            session.commit()
            self.today_trades += 1
            
            if not self.monitor.running:
                self.monitor.start()
                
        except Exception as e:
            logger.error(f"Execute signal error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def start_auto_mode(self, interval: int = 30):
        if self.auto_running:
            return
        self.auto_running = True
        self._auto_thread = threading.Thread(target=self._auto_loop, args=(interval,), daemon=True)
        self._auto_thread.start()
        self.monitor.start()
        logger.info(f"🔄 Auto mode started ({interval}s)")
    
    def stop_auto_mode(self):
        self.auto_running = False
        self.monitor.stop()
        logger.info("⏹️ Auto mode stopped")
    
    def _auto_loop(self, interval: int):
        while self.auto_running:
            try:
                config = self.get_config()
                if self.today_trades >= config['cs_max_daily_trades']:
                    self.status = "Daily limit reached"
                    time.sleep(60)
                    continue
                if config['cs_max_daily_loss'] > 0 and self.today_pnl <= -config['cs_max_daily_loss']:
                    self.status = "Daily loss limit"
                    time.sleep(60)
                    continue
                if not self.is_scanning:
                    self.start_scan()
                for _ in range(interval):
                    if not self.auto_running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Auto mode error: {e}")
                time.sleep(10)
    
    def get_signals(self, limit: int = 50) -> List[Dict]:
        session = db_manager.get_session()
        try:
            signals = session.query(ConfluenceSignal)\
                .order_by(ConfluenceSignal.created_at.desc()).limit(limit).all()
            return [{
                'id': s.id, 'symbol': s.symbol, 'direction': s.direction,
                'confluence_score': s.confluence_score, 'strength': s.strength,
                'entry_price': s.entry_price, 'current_price': s.current_price,
                'sl_price': s.sl_price, 'tp1_price': s.tp1_price, 'tp2_price': s.tp2_price,
                'status': s.status, 'pnl_percent': s.pnl_percent,
                'exit_reason': s.exit_reason, 'paper_trade': s.paper_trade,
                'tp1_hit': s.tp1_hit, 'problem_type': s.problem_type,
                'created_at': s.created_at.strftime('%d.%m %H:%M') if s.created_at else None,
            } for s in signals]
        finally:
            session.close()
    
    def get_open_positions(self) -> List[Dict]:
        session = db_manager.get_session()
        try:
            positions = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status.in_(['Open', 'TP1 Hit'])
            ).order_by(ConfluenceSignal.entry_time.desc()).all()
            result = []
            for p in positions:
                pnl = 0
                if p.current_price and p.entry_price:
                    if p.direction == 'LONG':
                        pnl = ((p.current_price - p.entry_price) / p.entry_price) * 100
                    else:
                        pnl = ((p.entry_price - p.current_price) / p.entry_price) * 100
                result.append({
                    'id': p.id, 'symbol': p.symbol, 'direction': p.direction,
                    'entry_price': p.entry_price, 'current_price': p.current_price,
                    'sl_price': p.sl_price, 'tp1_price': p.tp1_price, 'tp2_price': p.tp2_price,
                    'status': p.status, 'tp1_hit': p.tp1_hit,
                    'pnl_percent': round(pnl, 2),
                    'entry_time': p.entry_time.strftime('%H:%M') if p.entry_time else None,
                    'paper_trade': p.paper_trade,
                })
            return result
        finally:
            session.close()
    
    def get_scan_results(self) -> List[Dict]:
        return [r.to_dict() for r in self.scan_results]
    
    def get_stats(self) -> Dict:
        session = db_manager.get_session()
        try:
            all_signals = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status == 'Closed'
            ).all()
            if not all_signals:
                return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                       'total_pnl': 0, 'profit_factor': 0}
            wins = [s for s in all_signals if s.pnl_percent and s.pnl_percent > 0]
            losses = [s for s in all_signals if s.pnl_percent and s.pnl_percent <= 0]
            total_pnl = sum(s.pnl_percent or 0 for s in all_signals)
            gross_profit = sum(s.pnl_percent for s in wins) if wins else 0
            gross_loss = abs(sum(s.pnl_percent for s in losses)) if losses else 1
            return {
                'total_trades': len(all_signals),
                'wins': len(wins), 'losses': len(losses),
                'win_rate': round(len(wins) / len(all_signals) * 100, 1),
                'total_pnl': round(total_pnl, 2),
                'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            }
        finally:
            session.close()
    
    def get_analytics(self, days: int = 7) -> Dict:
        return self.analytics.analyze_performance(days)
    
    def clear_history(self):
        session = db_manager.get_session()
        try:
            session.query(ConfluenceSignal).delete()
            session.commit()
            self.today_trades = 0
            self.today_pnl = 0.0
            logger.info("🗑️ History cleared")
        except:
            session.rollback()
        finally:
            session.close()
    
    def close_position_manual(self, signal_id: int) -> Dict:
        session = db_manager.get_session()
        try:
            pos = session.query(ConfluenceSignal).filter_by(id=signal_id).first()
            if not pos or pos.status not in ['Open', 'TP1 Hit']:
                return {'status': 'error', 'error': 'Position not found or not open'}
            
            price = self.executor.get_current_price(pos.symbol)
            pos.exit_price = price
            pos.exit_time = datetime.utcnow()
            pos.exit_reason = 'Manual Close'
            pos.status = 'Closed'
            
            if pos.direction == "LONG":
                pos.pnl_percent = ((price - pos.entry_price) / pos.entry_price) * 100
            else:
                pos.pnl_percent = ((pos.entry_price - price) / pos.entry_price) * 100
            
            if not pos.paper_trade:
                self.executor.close_position(pos.symbol, pos.direction)
            
            session.commit()
            self.today_pnl += pos.pnl_percent
            return {'status': 'ok', 'pnl': pos.pnl_percent}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()


# ============================================================================
#                         GLOBAL INSTANCE
# ============================================================================

confluence_scalper = ConfluenceScalper()


# ============================================================================
#                          FLASK ROUTES
# ============================================================================

def register_routes(app):
    from flask import render_template, request, jsonify
    
    @app.route('/confluence_scalper')
    def confluence_scalper_page():
        return render_template(
            'confluence_scalper.html',
            config=confluence_scalper.get_config(),
            status=confluence_scalper.get_status(),
            presets=TIMEFRAME_PRESETS,
            help=PARAM_HELP,
            signals=confluence_scalper.get_signals(20),
            stats=confluence_scalper.get_stats(),
            open_positions=confluence_scalper.get_open_positions()
        )
    
    @app.route('/confluence_scalper/scan', methods=['POST'])
    def cs_scan():
        if confluence_scalper.start_scan():
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})
    
    @app.route('/confluence_scalper/stop', methods=['POST'])
    def cs_stop():
        confluence_scalper.stop_scan()
        return jsonify({'status': 'stopped'})
    
    @app.route('/confluence_scalper/status')
    def cs_status():
        return jsonify(confluence_scalper.get_status())
    
    @app.route('/confluence_scalper/results')
    def cs_results():
        return jsonify(confluence_scalper.get_scan_results())
    
    @app.route('/confluence_scalper/signals')
    def cs_signals():
        return jsonify(confluence_scalper.get_signals(50))
    
    @app.route('/confluence_scalper/positions')
    def cs_positions():
        return jsonify(confluence_scalper.get_open_positions())
    
    @app.route('/confluence_scalper/stats')
    def cs_stats():
        return jsonify(confluence_scalper.get_stats())
    
    @app.route('/confluence_scalper/analytics')
    def cs_analytics():
        days = request.args.get('days', 7, type=int)
        return jsonify(confluence_scalper.get_analytics(days))
    
    @app.route('/confluence_scalper/config', methods=['GET', 'POST'])
    def cs_config():
        if request.method == 'POST':
            data = request.json or {}
            settings.save_settings(data)
            
            # Handle auto_mode change
            if 'cs_auto_mode' in data:
                if data['cs_auto_mode']:
                    interval = data.get('cs_scan_interval', 30)
                    confluence_scalper.start_auto_mode(interval)
                else:
                    confluence_scalper.stop_auto_mode()
            
            return jsonify({'status': 'ok'})
        return jsonify(confluence_scalper.get_config())
    
    @app.route('/confluence_scalper/preset/<timeframe>')
    def cs_preset(timeframe):
        preset = TIMEFRAME_PRESETS.get(timeframe)
        if preset:
            return jsonify(preset)
        return jsonify({'error': 'Unknown timeframe'}), 404
    
    @app.route('/confluence_scalper/auto', methods=['POST'])
    def cs_auto():
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 30))
        if enabled:
            confluence_scalper.start_auto_mode(interval)
        else:
            confluence_scalper.stop_auto_mode()
        return jsonify({'status': 'ok', 'auto_running': confluence_scalper.auto_running})
    
    @app.route('/confluence_scalper/clear', methods=['POST'])
    def cs_clear():
        confluence_scalper.clear_history()
        return jsonify({'status': 'cleared'})
    
    @app.route('/confluence_scalper/close/<int:signal_id>', methods=['POST'])
    def cs_close_position(signal_id):
        return jsonify(confluence_scalper.close_position_manual(signal_id))
    
    @app.route('/confluence_scalper/execute/<int:index>', methods=['POST'])
    def cs_execute_signal(index):
        if index < 0 or index >= len(confluence_scalper.scan_results):
            return jsonify({'status': 'error', 'error': 'Invalid index'})
        signal = confluence_scalper.scan_results[index]
        config = confluence_scalper.get_config()
        confluence_scalper._execute_signal(signal, config)
        return jsonify({'status': 'ok'})
    
    logger.info("✅ Confluence Scalper v2.0 routes registered")
