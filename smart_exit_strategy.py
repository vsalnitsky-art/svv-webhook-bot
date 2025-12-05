#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

class SmartExitStrategy:
    """
    Професійна стратегія Smart Exit на основі:
    1. Trailing Stop (-0.5% від fresh_max)
    2. Дивергенція RSI (ціна ↑ але RSI ↓)
    """
    
    def __init__(self):
        # Зберігаємо стан для кожної позиції
        self.position_states = defaultdict(lambda: {
            'mode': 'WAITING',              # WAITING або HUNTING
            'entry_price': None,
            'fresh_max_price': None,
            'trailing_stop': None,
            'rsi_activated_at': None,
            'prev_rsi': None,
            'curr_rsi': None,
            'prev_price': None,
            'curr_price': None,
            'divergence_count': 0,          # Лічильник дивергенцій
            'activated_timestamp': None,
            'exit_reason': None
        })
        
        self.TRAILING_STOP_PERCENT = -0.005  # -0.5%
        self.MIN_DIVERGENCE_CANDLES = 2      # Мінімум 2 свічки дивергенції
        self.RSI_THRESHOLD = 70              # Для LONG
        
    def update_position(self, symbol, current_price, rsi_value, side='Long'):
        """
        Оновлюємо стан позиції та перевіряємо умови закриття
        
        Args:
            symbol: BTCUSDT
            current_price: поточна ціна
            rsi_value: поточний RSI
            side: 'Long' або 'Short'
        
        Returns:
            {
                'should_close': True/False,
                'reason': 'Trailing Stop' або 'Divergence' або 'None',
                'exit_price': число,
                'profit_potential': число
            }
        """
        
        state = self.position_states[symbol]
        
        # ✅ ФАЗА 1: ОЧІКУВАННЯ
        if state['mode'] == 'WAITING':
            # Перевіряємо, чи включити режим HUNTING
            if self._should_activate_hunting(rsi_value, side):
                logger.info(f"🎯 ACTIVATE HUNTING MODE: {symbol} | RSI={rsi_value}")
                
                state['mode'] = 'HUNTING'
                state['entry_price'] = current_price
                state['fresh_max_price'] = current_price
                state['trailing_stop'] = current_price * (1 + self.TRAILING_STOP_PERCENT)
                state['rsi_activated_at'] = rsi_value
                state['prev_rsi'] = rsi_value
                state['curr_rsi'] = rsi_value
                state['prev_price'] = current_price
                state['curr_price'] = current_price
                state['divergence_count'] = 0
                state['activated_timestamp'] = datetime.now()
                
                return {
                    'should_close': False,
                    'reason': 'Mode activated, waiting for signals',
                    'exit_price': None,
                    'profit_potential': 0
                }
        
        # ✅ ФАЗА 2: ЛОВЛЯ МАКСИМУМУ
        elif state['mode'] == 'HUNTING':
            
            # Оновлюємо попередні значення
            state['prev_rsi'] = state['curr_rsi']
            state['prev_price'] = state['curr_price']
            state['curr_rsi'] = rsi_value
            state['curr_price'] = current_price
            
            # ПЕРЕВІРКА 1: Чи оновився максимум?
            if current_price > state['fresh_max_price']:
                state['fresh_max_price'] = current_price
                state['trailing_stop'] = current_price * (1 + self.TRAILING_STOP_PERCENT)
                
                logger.debug(f"📈 Fresh max updated: {symbol} | Price={current_price} | TS={state['trailing_stop']:.2f}")
                
                # Обнулюємо дивергенцію якщо ціна робить новий max
                state['divergence_count'] = 0
            
            # ПЕРЕВІРКА 2: Trailing Stop сработав?
            if self._check_trailing_stop(state, current_price):
                state['mode'] = 'CLOSED'
                state['exit_reason'] = 'Trailing Stop'
                
                profit = ((current_price - state['entry_price']) / state['entry_price']) * 100
                
                logger.info(
                    f"🛑 TRAILING STOP HIT: {symbol}\n"
                    f"   Entry: {state['entry_price']:.2f}\n"
                    f"   Exit: {current_price:.2f}\n"
                    f"   Max touched: {state['fresh_max_price']:.2f}\n"
                    f"   Profit: {profit:.2f}%"
                )
                
                return {
                    'should_close': True,
                    'reason': 'Trailing Stop',
                    'exit_price': current_price,
                    'profit_potential': profit
                }
            
            # ПЕРЕВІРКА 3: Дивергенція?
            if self._check_divergence(state, current_price, rsi_value):
                state['divergence_count'] += 1
                logger.debug(f"⚠️ Divergence detected: {symbol} | Count={state['divergence_count']}")
                
                # Якщо дивергенція тривала 2+ свічки - закриваємо
                if state['divergence_count'] >= self.MIN_DIVERGENCE_CANDLES:
                    state['mode'] = 'CLOSED'
                    state['exit_reason'] = 'Divergence'
                    
                    profit = ((current_price - state['entry_price']) / state['entry_price']) * 100
                    
                    logger.info(
                        f"📉 DIVERGENCE EXIT: {symbol}\n"
                        f"   Entry: {state['entry_price']:.2f}\n"
                        f"   Exit: {current_price:.2f}\n"
                        f"   Max touched: {state['fresh_max_price']:.2f}\n"
                        f"   Divergence candles: {state['divergence_count']}\n"
                        f"   Profit: {profit:.2f}%"
                    )
                    
                    return {
                        'should_close': True,
                        'reason': 'Divergence',
                        'exit_price': current_price,
                        'profit_potential': profit
                    }
            else:
                # Дивергенція исчезла - обнулюємо лічильник
                state['divergence_count'] = 0
        
        # Немає сигналів на закриття
        return {
            'should_close': False,
            'reason': 'No exit signal',
            'exit_price': None,
            'profit_potential': None
        }
    
    def _should_activate_hunting(self, rsi_value, side='Long'):
        """Чи включити режим HUNTING?"""
        # Для LONG: RSI >= 70 (перекупленість)
        # Для SHORT: RSI <= 30 (перепроданість)
        
        if side == 'Long':
            return rsi_value >= self.RSI_THRESHOLD
        else:
            return rsi_value <= (100 - self.RSI_THRESHOLD)
    
    def _check_trailing_stop(self, state, current_price):
        """
        Перевіряємо: Чи ціна впала нижче trailing stop?
        
        Trailing Stop = fresh_max * (1 - 0.5%) = fresh_max * 0.995
        
        Якщо current_price < trailing_stop → CLOSE
        """
        
        if state['trailing_stop'] is None:
            return False
        
        result = current_price < state['trailing_stop']
        
        if result:
            logger.debug(
                f"⚠️ Trailing Stop triggered:\n"
                f"   Current: {current_price:.2f}\n"
                f"   Stop: {state['trailing_stop']:.2f}\n"
                f"   Diff: {(state['trailing_stop'] - current_price):.2f}"
            )
        
        return result
    
    def _check_divergence(self, state, current_price, rsi_value):
        """
        Перевіряємо дивергенцію:
        - Ціна росте (або стоїть) нові max
        - Але RSI падає
        
        Дивергенція = Price вверх И RSI вниз одночасно
        """
        
        if state['prev_price'] is None or state['prev_rsi'] is None:
            return False
        
        # Умова: ціна не впала нижче fresh_max, але RSI впав
        price_not_falling = current_price >= (state['fresh_max_price'] * 0.99)  # Дозволяємо -1% шум
        rsi_falling = rsi_value < state['prev_rsi']
        
        divergence_detected = price_not_falling and rsi_falling
        
        if divergence_detected:
            logger.debug(
                f"🔄 Divergence check:\n"
                f"   Price: {state['prev_price']:.2f} → {current_price:.2f}\n"
                f"   RSI: {state['prev_rsi']:.1f} → {rsi_value:.1f}\n"
                f"   Fresh Max: {state['fresh_max_price']:.2f}\n"
                f"   Result: DIVERGENCE"
            )
        
        return divergence_detected
    
    def reset_position(self, symbol):
        """Скидаємо стан позиції (після закриття)"""
        self.position_states[symbol] = {
            'mode': 'WAITING',
            'entry_price': None,
            'fresh_max_price': None,
            'trailing_stop': None,
            'rsi_activated_at': None,
            'prev_rsi': None,
            'curr_rsi': None,
            'prev_price': None,
            'curr_price': None,
            'divergence_count': 0,
            'activated_timestamp': None,
            'exit_reason': None
        }
        
        logger.info(f"🔄 Position state reset for {symbol}")
    
    def get_position_status(self, symbol):
        """Отримуємо поточний стан позиції (для debugging)"""
        state = self.position_states[symbol]
        
        if state['mode'] == 'WAITING':
            return {
                'status': 'WAITING for RSI >= 70',
                'details': None
            }
        
        elif state['mode'] == 'HUNTING':
            return {
                'status': 'HUNTING maximum',
                'details': {
                    'entry_price': state['entry_price'],
                    'fresh_max_price': state['fresh_max_price'],
                    'trailing_stop': state['trailing_stop'],
                    'rsi_activated_at': state['rsi_activated_at'],
                    'divergence_count': state['divergence_count'],
                    'time_active': (datetime.now() - state['activated_timestamp']).total_seconds() if state['activated_timestamp'] else 0
                }
            }
        
        else:
            return {
                'status': 'CLOSED',
                'reason': state['exit_reason'],
                'details': None
            }


# 🎯 ГЛОБАЛЬНИЙ ЕКЗЕМПЛЯР
smart_exit = SmartExitStrategy()
