"""
Enhanced Market Scanner v2.0
Professional RSI+MFI based scanner with auto-close and market scanning
OBV integration for volume confirmation
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from rsi_mfi_indicator import RSIMFIIndicator
from obv_indicator import OBVIndicator  # ⭐ НОВОЕ
from scanner_config import ScannerConfig
from models import (db_manager, PositionAnalytics, PositionSnapshot, 
                    MarketCandidate, ScanHistory, AutoCloseDecision)
from market_scanner import MarketScanner

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Мониторинг открытых позиций и автоматическое закрытие
    """
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.bot = scanner.bot
        self.config = scanner.config
        self.indicator = scanner.indicator
        
        # OBV индикатор (НОВОЕ) ⭐
        obv_period = self.config.auto_close_params.get('obv_ema_period', 20)
        self.obv_indicator = OBVIndicator(ema_period=obv_period)
        
        # Хранилище данных по активным позициям
        self.active_positions: Dict[str, Dict] = {}
        
        # Состояния мониторинга RSI зон (НОВОЕ) ⭐
        self.rsi_zone_states: Dict[str, Dict] = {}
        # Формат: {
        #     'BTCUSDT': {
        #         'in_zone': True/False,
        #         'entry_time': datetime,
        #         'obv_monitoring': True/False,
        #         'last_obv_check': datetime
        #     }
        # }
        
        # Статистика
        self.last_monitor_time = None
        self.total_auto_closes = 0
        self.successful_auto_closes = 0
        
        logger.info("✅ PositionMonitor ініціалізовано з підтримкою OBV")
    
    def start(self):
        """Запуск мониторинга в отдельном потоке"""
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        logger.info("🚀 PositionMonitor запущено")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while True:
            try:
                self.monitor_positions()
                time.sleep(5)  # Проверка каждые 5 секунд
            except Exception as e:
                logger.error(f"❌ Помилка циклу моніторингу: {e}", exc_info=True)
                time.sleep(10)
    
    def monitor_positions(self):
        """
        Основная функция мониторинга позиций
        """
        try:
            # Получить список активных позиций с биржи
            positions = self._get_active_positions()
            
            if not positions:
                # Нет активных позиций - очистить память
                self.active_positions = {}
                self.last_monitor_time = datetime.now()
                return
            
            # Обработать каждую позицию
            for position in positions:
                symbol = position['symbol']
                
                try:
                    # Получить исторические данные для индикатора
                    df = self._get_historical_data(symbol)
                    
                    if df is None or len(df) < 50:
                        logger.warning(f"⚠️ Insufficient data for {symbol}")
                        continue
                    
                    # Применить индикатор
                    result_df, alerts = self.indicator.process_data(df, symbol)
                    
                    # Получить последние сигналы
                    latest_signals = self.indicator.get_latest_signals(result_df)
                    
                    # Обновить статистику позиции
                    self._update_position_stats(symbol, position, latest_signals)
                    
                    # Сохранить снимок состояния
                    self._save_position_snapshot(symbol, position, latest_signals)
                    
                    # Проверить условия автозакрытия
                    if self.config.get_auto_close_params()['enabled']:
                        self._check_auto_close(symbol, position, latest_signals)
                    
                except Exception as e:
                    logger.error(f"❌ Помилка моніторингу {symbol}: {e}")
                    continue
            
            self.last_monitor_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Monitor positions error: {e}", exc_info=True)
    
    def _get_active_positions(self) -> List[Dict]:
        """
        Получить список активных позиций с биржи
        """
        try:
            response = self.bot.session.get_positions(
                category="linear", 
                settleCoin="USDT"
            )
            
            if response['retCode'] != 0:
                logger.error(f"❌ API error: {response.get('retMsg')}")
                return []
            
            # Фильтровать только открытые позиции
            positions = []
            for pos in response['result']['list']:
                if float(pos['size']) > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],  # Buy=Long, Sell=Short
                        'size': float(pos['size']),
                        'entry_price': float(pos['avgPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealised_pnl': float(pos['unrealisedPnl']),
                        'leverage': pos['leverage'],
                        'stop_loss': pos.get('stopLoss', None),
                        'take_profit': pos.get('takeProfit', None),
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def _get_historical_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Получить исторические данные для индикатора
        Использует таймфрейм из конфигурации (по умолчанию 4h)
        """
        try:
            # Получить таймфрейм из конфигурации
            timeframe = self.scanner.config.get_timeframe()
            
            # Получить klines с биржи
            response = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,  # ✅ Использует timeframe из конфига
                limit=limit
            )
            
            if response['retCode'] != 0:
                logger.error(f"❌ Kline API error for {symbol}: {response.get('retMsg')}")
                return None
            
            klines = response['result']['list']
            
            if not klines:
                return None
            
            # Преобразовать в DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Конвертировать типы
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Установить timestamp как индекс
            df.set_index('timestamp', inplace=True)
            
            # Bybit возвращает в обратном порядке, нужно отсортировать
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return None
    
    def _update_position_stats(self, symbol: str, position: Dict, signals: Dict):
        """
        Обновить статистику по позиции
        """
        if symbol not in self.active_positions:
            # Первая запись - инициализация
            self.active_positions[symbol] = {
                'open_time': datetime.now(),
                'entry_price': position['entry_price'],
                'side': position['side'],
                'max_pnl': position['unrealised_pnl'],
                'min_pnl': position['unrealised_pnl'],
                'rsi_values': [],
                'mfi_values': [],
                'signal_count': 0,
                'last_signal': None,
            }
        
        stats = self.active_positions[symbol]
        
        # Обновить максимальный и минимальный P&L
        current_pnl = position['unrealised_pnl']
        stats['max_pnl'] = max(stats['max_pnl'], current_pnl)
        stats['min_pnl'] = min(stats['min_pnl'], current_pnl)
        
        # Сохранить значения индикаторов
        stats['rsi_values'].append(signals['rsi_value'])
        stats['mfi_values'].append(signals.get('mfi_value', 0))
        
        # Ограничить размер истории (последние 100 значений)
        if len(stats['rsi_values']) > 100:
            stats['rsi_values'] = stats['rsi_values'][-100:]
            stats['mfi_values'] = stats['mfi_values'][-100:]
        
        # Подсчитать сигналы
        if signals['last_signal'] and signals['last_signal'] != 'Нет':
            if signals['last_signal'] != stats['last_signal']:
                stats['signal_count'] += 1
                stats['last_signal'] = signals['last_signal']
    
    def _save_position_snapshot(self, symbol: str, position: Dict, signals: Dict):
        """
        Сохранить снимок состояния позиции в БД
        """
        try:
            session = db_manager.get_session()
            
            snapshot = PositionSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                price=position['mark_price'],
                pnl=position['unrealised_pnl'],
                rsi=signals['rsi_value'],
                mfi=signals.get('mfi_value', 0),
                mfi_trend=signals.get('mfi_trend', 'Нейтральный'),
                current_signal=signals.get('last_signal', 'Нет'),
                momentum=self._calculate_momentum(signals)
            )
            
            session.add(snapshot)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving snapshot for {symbol}: {e}")
    
    def _calculate_momentum(self, signals: Dict) -> str:
        """
        Определить моментум (Растёт/Падает/Флэт)
        """
        rsi = signals['rsi_value']
        
        # Упрощённая логика на основе RSI
        if rsi > 55:
            return "Растёт"
        elif rsi < 45:
            return "Падает"
        else:
            return "Флэт"
    
    def _check_auto_close(self, symbol: str, position: Dict, signals: Dict):
        """
        Проверить условия автозакрытия позиции
        """
        auto_close_config = self.config.get_auto_close_params()
        
        # Проверка минимального времени удержания
        if not self._check_min_hold_time(symbol, auto_close_config):
            return
        
        # Определить направление позиции
        side = position['side']  # Buy=Long, Sell=Short
        is_long = (side == 'Buy')
        is_short = (side == 'Sell')
        
        # Текущий P&L
        current_pnl = position['unrealised_pnl']
        
        # Добавить symbol в signals для OBV логики ⭐
        signals['symbol'] = symbol
        
        # Проверить условия закрытия
        should_close, reason = self._evaluate_close_conditions(
            is_long, is_short, signals, current_pnl, auto_close_config
        )
        
        # Логировать решение
        self._log_close_decision(symbol, should_close, reason, signals, current_pnl)
        
        # Выполнить закрытие если нужно
        if should_close:
            self._execute_auto_close(symbol, position, reason)
    
    def _check_min_hold_time(self, symbol: str, config: Dict) -> bool:
        """
        Проверить минимальное время удержания позиции
        """
        if symbol not in self.active_positions:
            return False
        
        open_time = self.active_positions[symbol]['open_time']
        hold_duration = (datetime.now() - open_time).total_seconds()
        min_hold_time = config['min_hold_time']
        
        if hold_duration < min_hold_time:
            return False
        
        return True
    
    def _evaluate_close_conditions(
        self, 
        is_long: bool, 
        is_short: bool, 
        signals: Dict, 
        current_pnl: float,
        config: Dict
    ) -> Tuple[bool, str]:
        """
        Оценить условия закрытия с OBV подтверждением
        
        Новая логика:
        1. RSI входит в зону → запускаем мониторинг OBV
        2. Проверяем OBV тренд (если включено)
        3. Проверяем выход из зоны RSI
        4. MFI подтверждение (если включено и в соотв. с режимом)
        
        Returns:
            (should_close: bool, reason: str)
        """
        
        # Извлечь параметры
        use_strong = config['use_strong_signals']
        use_obv = config.get('use_obv_confirmation', True)
        obv_sensitivity = config.get('obv_sensitivity', 'high')
        obv_trend_candles = config.get('obv_trend_candles', 3)
        rsi_exit_mode = config.get('rsi_exit_mode', 'wait_zone_exit')
        confirm_mfi = config.get('confirm_with_mfi', False)
        mfi_check_mode = config.get('mfi_check_mode', 'after_obv')
        
        rsi = signals['rsi_value']
        mfi_trend = signals.get('mfi_trend', 'Нейтральный')
        symbol = signals.get('symbol', 'UNKNOWN')
        
        # Инициализировать состояние если нужно
        if symbol not in self.rsi_zone_states:
            self.rsi_zone_states[symbol] = {
                'in_zone': False,
                'entry_time': None,
                'obv_monitoring': False,
                'last_obv_check': None
            }
        
        state = self.rsi_zone_states[symbol]
        
        # === ЛОГИКА ДЛЯ LONG ПОЗИЦИЙ ===
        if is_long:
            overbought_threshold = 70  # Порог перекупленности
            
            # Проверка: RSI в зоне перекупленности?
            in_overbought = (rsi > overbought_threshold)
            
            if in_overbought and not state['in_zone']:
                # ВОШЛИ В ЗОНУ ПЕРЕКУПЛЕННОСТИ
                state['in_zone'] = True
                state['entry_time'] = datetime.now()
                state['obv_monitoring'] = use_obv
                logger.info(f"📊 {symbol}: RSI увійшов у зону перекупленості ({rsi:.1f})")
                
                # Если OBV выключен и режим immediate
                if not use_obv and rsi_exit_mode == 'immediate':
                    return True, f"RSI overbought ({rsi:.1f}) - immediate exit mode"
            
            if state['in_zone']:
                # УЖЕ В ЗОНЕ - проверяем условия выхода
                
                # MFI проверка ПЕРЕД OBV (если режим before_obv)
                if confirm_mfi and mfi_check_mode == 'before_obv':
                    if mfi_trend != 'Медвежий':
                        return False, f"Waiting for bearish MFI (current: {mfi_trend})"
                
                # Проверка OBV (если включено)
                if use_obv and state['obv_monitoring']:
                    # Получить исторические данные
                    candles = self._get_historical_data_for_obv(symbol)
                    
                    if candles and len(candles) > 20:
                        # Проверить OBV тренд
                        should_close_obv, reason_obv, details = self.obv_indicator.check_trend(
                            candles=candles,
                            direction='down',  # Long → ищем падение OBV
                            n_candles=obv_trend_candles,
                            sensitivity=obv_sensitivity
                        )
                        
                        state['last_obv_check'] = datetime.now()
                        
                        if should_close_obv:
                            # OBV ПОДТВЕРДИЛ РАЗВОРОТ!
                            logger.info(f"✅ {symbol}: OBV підтвердив розворот - {reason_obv}")
                            
                            # MFI проверка ПОСЛЕ OBV (если режим after_obv)
                            if confirm_mfi and mfi_check_mode == 'after_obv':
                                if mfi_trend == 'Медвежий':
                                    return True, f"RSI overbought + {reason_obv} + bearish MFI"
                                else:
                                    logger.info(f"⚠️ {symbol}: OBV confirmed but waiting for bearish MFI")
                                    return False, f"OBV confirmed, waiting for MFI (current: {mfi_trend})"
                            
                            # MFI проверка ВМЕСТЕ С OBV (если режим with_obv)
                            if confirm_mfi and mfi_check_mode == 'with_obv':
                                if mfi_trend == 'Медвежий':
                                    return True, f"RSI overbought + {reason_obv} + bearish MFI"
                                else:
                                    return False, f"OBV confirmed but MFI not bearish"
                            
                            # MFI не требуется
                            return True, f"RSI overbought + {reason_obv}"
                        else:
                            # OBV ещё не подтвердил
                            logger.debug(f"⏳ {symbol}: Моніторинг OBV - {reason_obv}")
                
                # Проверка: ВЫШЛИ ИЗ ЗОНЫ?
                if not in_overbought:
                    # RSI вышел из зоны перекупленности
                    logger.info(f"📉 {symbol}: RSI вийшов із зони перекупленості ({rsi:.1f})")
                    state['in_zone'] = False
                    state['obv_monitoring'] = False
                    
                    # MFI проверка при выходе
                    if confirm_mfi:
                        if mfi_trend == 'Медвежий':
                            return True, f"RSI exited overbought ({rsi:.1f}) + bearish MFI"
                        elif mfi_check_mode != 'with_obv':
                            return True, f"RSI exited overbought ({rsi:.1f})"
                    else:
                        return True, f"RSI вийшов із зони перекупленості ({rsi:.1f})"
        
        # === ЛОГИКА ДЛЯ SHORT ПОЗИЦИЙ ===
        if is_short:
            oversold_threshold = 30  # Порог перепроданности
            
            # Проверка: RSI в зоне перепроданности?
            in_oversold = (rsi < oversold_threshold)
            
            if in_oversold and not state['in_zone']:
                # ВОШЛИ В ЗОНУ ПЕРЕПРОДАННОСТИ
                state['in_zone'] = True
                state['entry_time'] = datetime.now()
                state['obv_monitoring'] = use_obv
                logger.info(f"📊 {symbol}: RSI увійшов у зону перепроданості ({rsi:.1f})")
                
                # Если OBV выключен и режим immediate
                if not use_obv and rsi_exit_mode == 'immediate':
                    return True, f"RSI oversold ({rsi:.1f}) - immediate exit mode"
            
            if state['in_zone']:
                # УЖЕ В ЗОНЕ - проверяем условия выхода
                
                # MFI проверка ПЕРЕД OBV
                if confirm_mfi and mfi_check_mode == 'before_obv':
                    if mfi_trend != 'Бычий':
                        return False, f"Waiting for bullish MFI (current: {mfi_trend})"
                
                # Проверка OBV (если включено)
                if use_obv and state['obv_monitoring']:
                    candles = self._get_historical_data_for_obv(symbol)
                    
                    if candles and len(candles) > 20:
                        should_close_obv, reason_obv, details = self.obv_indicator.check_trend(
                            candles=candles,
                            direction='up',  # Short → ищем рост OBV
                            n_candles=obv_trend_candles,
                            sensitivity=obv_sensitivity
                        )
                        
                        state['last_obv_check'] = datetime.now()
                        
                        if should_close_obv:
                            # OBV ПОДТВЕРДИЛ РАЗВОРОТ!
                            logger.info(f"✅ {symbol}: OBV підтвердив розворот - {reason_obv}")
                            
                            # MFI проверка ПОСЛЕ OBV
                            if confirm_mfi and mfi_check_mode == 'after_obv':
                                if mfi_trend == 'Бычий':
                                    return True, f"RSI oversold + {reason_obv} + bullish MFI"
                                else:
                                    return False, f"OBV confirmed, waiting for MFI"
                            
                            # MFI проверка ВМЕСТЕ С OBV
                            if confirm_mfi and mfi_check_mode == 'with_obv':
                                if mfi_trend == 'Бычий':
                                    return True, f"RSI oversold + {reason_obv} + bullish MFI"
                                else:
                                    return False, f"OBV confirmed but MFI not bullish"
                            
                            return True, f"RSI oversold + {reason_obv}"
                
                # Проверка: ВЫШЛИ ИЗ ЗОНЫ?
                if not in_oversold:
                    logger.info(f"📈 {symbol}: RSI вийшов із зони перепроданості ({rsi:.1f})")
                    state['in_zone'] = False
                    state['obv_monitoring'] = False
                    
                    # MFI проверка при выходе
                    if confirm_mfi:
                        if mfi_trend == 'Бычий':
                            return True, f"RSI exited oversold ({rsi:.1f}) + bullish MFI"
                        elif mfi_check_mode != 'with_obv':
                            return True, f"RSI exited oversold ({rsi:.1f})"
                    else:
                        return True, f"RSI вийшов із зони перепроданості ({rsi:.1f})"
        
        # Нет условий для закрытия
        return False, "Умови закриття не виконано"
    
    def _get_historical_data_for_obv(self, symbol: str) -> List[Dict]:
        """
        Получить исторические данные для OBV расчёта
        
        Returns:
            List of candles with close, volume
        """
        try:
            # Получить таймфрейм из конфигурации
            timeframe = self.scanner.config.get_timeframe()
            
            # Нужно достаточно свечей для EMA + проверки тренда
            limit = self.config.auto_close_params.get('obv_ema_period', 20) + 20
            
            response = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if response['retCode'] == 0:
                candles_data = response['result']['list']
                
                # Преобразовать в нужный формат
                candles = []
                for candle in reversed(candles_data):  # Bybit возвращает в обратном порядке
                    candles.append({
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'timestamp': int(candle[0])
                    })
                
                return candles
            else:
                logger.error(f"Failed to get candles for {symbol}: {response.get('retMsg')}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting historical data for OBV: {e}")
            return []
    
    def _log_close_decision(
        self, 
        symbol: str, 
        decision: bool, 
        reason: str, 
        signals: Dict, 
        pnl: float
    ):
        """
        Логировать решение о закрытии в БД
        """
        if not self.config.get_auto_close_params()['log_all_decisions']:
            return
        
        try:
            session = db_manager.get_session()
            
            log = AutoCloseDecision(
                symbol=symbol,
                timestamp=datetime.now(),
                decision='close' if decision else 'hold',
                reason=reason,
                rsi=signals['rsi_value'],
                mfi=signals.get('mfi_value', 0),
                signal=signals.get('last_signal', 'Нет'),
                pnl_at_decision=pnl,
                executed=False  # Будет обновлено после выполнения
            )
            
            session.add(log)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"❌ Error logging decision for {symbol}: {e}")
    
    def _execute_auto_close(self, symbol: str, position: Dict, reason: str):
        """
        Выполнить автоматическое закрытие позиции
        """
        try:
            side = position['side']
            direction = 'Long' if side == 'Buy' else 'Short'
            
            logger.info(f"🔴 AUTO-CLOSE: {symbol} {direction} | Reason: {reason}")
            
            # Отправить команду закрытия в bot
            result = self.bot.place_order({
                'action': 'Close',
                'symbol': symbol,
                'direction': direction,
                'reason': f"Auto-close: {reason}"
            })
            
            if result.get('status') == 'ok':
                logger.info(f"✅ Позицію закрито: {symbol}")
                self.successful_auto_closes += 1
                
                # Сохранить аналитику закрытия
                self._save_close_analytics(symbol, position, reason, success=True)
                
                # Удалить из активных позиций
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                
            else:
                logger.error(f"❌ Failed to close {symbol}: {result}")
                self._save_close_analytics(symbol, position, reason, success=False)
            
            self.total_auto_closes += 1
            
        except Exception as e:
            logger.error(f"❌ Error executing auto-close for {symbol}: {e}")
    
    def _save_close_analytics(
        self, 
        symbol: str, 
        position: Dict, 
        reason: str, 
        success: bool
    ):
        """
        Сохранить детальную аналитику закрытой позиции
        """
        if not self.config.get_auto_close_params()['save_close_analytics']:
            return
        
        try:
            stats = self.active_positions.get(symbol, {})
            
            if not stats:
                return
            
            session = db_manager.get_session()
            
            # Рассчитать метрики
            open_time = stats.get('open_time', datetime.now())
            hold_duration = (datetime.now() - open_time).total_seconds()
            
            avg_rsi = np.mean(stats['rsi_values']) if stats['rsi_values'] else 0
            rsi_range = f"{min(stats['rsi_values']):.1f}-{max(stats['rsi_values']):.1f}" if stats['rsi_values'] else "N/A"
            
            avg_mfi = np.mean(stats['mfi_values']) if stats['mfi_values'] else 0
            
            entry_price = stats.get('entry_price', position['entry_price'])
            exit_price = position['mark_price']
            pnl = position['unrealised_pnl']
            pnl_percent = (pnl / (entry_price * position['size'])) * 100 if position['size'] > 0 else 0
            
            analytics = PositionAnalytics(
                symbol=symbol,
                side='Long' if position['side'] == 'Buy' else 'Short',
                open_time=open_time,
                close_time=datetime.now(),
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                max_pnl=stats.get('max_pnl', pnl),
                min_pnl=stats.get('min_pnl', pnl),
                avg_rsi=avg_rsi,
                rsi_range=rsi_range,
                avg_mfi=avg_mfi,
                mfi_trend='N/A',  # Можно улучшить
                signal_count=stats.get('signal_count', 0),
                hold_duration=int(hold_duration),
                close_reason=reason,
                close_method='auto' if success else 'auto_failed'
            )
            
            session.add(analytics)
            session.commit()
            session.close()
            
            logger.info(f"📊 Analytics saved for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error saving analytics for {symbol}: {e}")
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """
        Получить информацию о позиции для UI
        """
        if symbol not in self.active_positions:
            return None
        
        stats = self.active_positions[symbol]
        
        return {
            'open_time': stats['open_time'],
            'hold_duration': (datetime.now() - stats['open_time']).total_seconds(),
            'max_pnl': stats['max_pnl'],
            'min_pnl': stats['min_pnl'],
            'avg_rsi': np.mean(stats['rsi_values']) if stats['rsi_values'] else 0,
            'rsi_range': f"{min(stats['rsi_values']):.1f}-{max(stats['rsi_values']):.1f}" if stats['rsi_values'] else "N/A",
            'avg_mfi': np.mean(stats['mfi_values']) if stats['mfi_values'] else 0,
            'signal_count': stats['signal_count'],
            'last_signal': stats['last_signal'],
        }
    
    def get_stats(self) -> Dict:
        """
        Получить статистику мониторинга
        """
        return {
            'active_positions': len(self.active_positions),
            'total_auto_closes': self.total_auto_closes,
            'successful_auto_closes': self.successful_auto_closes,
            'success_rate': (self.successful_auto_closes / self.total_auto_closes * 100) 
                           if self.total_auto_closes > 0 else 0,
            'last_monitor_time': self.last_monitor_time,
        }


class EnhancedMarketScanner:
    """
    Главный класс сканера - координатор всех компонентов
    """
    
    def __init__(self, bot_instance, config: ScannerConfig):
        self.bot = bot_instance
        self.config = config
        
        # Создать индикатор с параметрами из конфигурации
        self.indicator = RSIMFIIndicator(**config.get_indicator_params())
        
        # Создать компоненты
        self.position_monitor = PositionMonitor(self)
        self.market_scanner = MarketScanner(self)  # ✅ Добавлен MarketScanner
        
        # Временные данные для совместимости с UI
        self.active_coins_data = {}
        self.last_scan_time = None
        
        logger.info("✅ EnhancedMarketScanner v2.0 initialized")
    
    def start(self):
        """Запуск всех компонентов"""
        self.position_monitor.start()
        self.market_scanner.start()  # ✅ Запускаем сканирование рынка
        logger.info("🚀 Scanner v2.0 started")
    
    def get_current_rsi(self, symbol: str) -> float:
        """Получить текущий RSI для UI (совместимость)"""
        # TODO: Получать из PositionMonitor
        return 50.0
    
    def get_market_pressure(self, symbol: str) -> float:
        """Получить рыночное давление для UI (совместимость)"""
        # TODO: Рассчитывать из volume data
        return 0.0
    
    def get_aggregated_data(self, hours: int = 24) -> Dict:
        """Получить агрегированные данные для UI"""
        return {
            'all_signals': [],
            'last_scan': self.last_scan_time,
            'snapshots': self.active_coins_data,
            'monitor_stats': self.position_monitor.get_stats(),
            'scanner_stats': self.market_scanner.get_stats(),  # ✅ Добавлена статистика сканера
        }
