#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 ТАЙМФРЕЙМ-ЗАЛЕЖНІ ПОКАЗНИКИ
Професійна система автоматичного налаштування параметрів по таймфрейму
"""

import logging

logger = logging.getLogger(__name__)

class TimeframeParameters:
    """
    Професійні показники для кожного таймфрейму та профілю
    
    Таймфрейми:
    - 1m  - Скальпінг (дуже короткий)
    - 5m  - Скальпінг (короткий)
    - 15m - Короткостроковий трейдинг
    - 1h  - Середньостроковий (СТАНДАРТ)
    - 4h  - Середньостроковий+ 
    - 1d  - Довгостроковий
    """
    
    # Таблиця параметрів: timeframe -> { profile -> { параметри } }
    
    TIMEFRAME_CONFIGS = {
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 1m - СКАЛЬПІНГ ДУЖЕ КОРОТКИЙ
        # ═══════════════════════════════════════════════════════════════════════════
        '1m': {
            'rsi_period': 7,              # Коротка довжина для швидких ходів
            'rsi_threshold_long': 75,     # Вищий поріг (дуже перекуплено)
            'rsi_threshold_short': 25,    # Нижче
            'trailing_stop_percent': -0.012,  # -1.2% (більший для швидких дропів)
            'min_divergence_candles': 1,  # 1 свічка достатньо
            'stop_loss_percent': 1.5,     # 1.5% SL
            'take_profit_percent': 0.5,   # 0.5% TP
            'hma_fast_period': 5,         # Швидкий MA
            'hma_slow_period': 10,        # Повільний MA
            'expected_return': '+0.3% - +0.6%',
            'expected_max_loss': '-1.2%',
            'strategy_type': 'SCALPING',
            'trades_per_day': '50-100',
            'hold_time': '1-10 minutes'
        },
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 5m - СКАЛЬПІНГ КОРОТКИЙ
        # ═══════════════════════════════════════════════════════════════════════════
        '5m': {
            'rsi_period': 9,              # Коротка для швидкого таймфрейму
            'rsi_threshold_long': 73,     # Перекупленість
            'rsi_threshold_short': 27,    # Перепроданість
            'trailing_stop_percent': -0.010,  # -1.0%
            'min_divergence_candles': 1,  # 1 свічка
            'stop_loss_percent': 1.2,     # 1.2% SL
            'take_profit_percent': 0.8,   # 0.8% TP
            'hma_fast_period': 7,         # Швидкий MA
            'hma_slow_period': 14,        # Повільний MA
            'expected_return': '+0.5% - +1.0%',
            'expected_max_loss': '-1.0%',
            'strategy_type': 'SCALPING',
            'trades_per_day': '30-50',
            'hold_time': '5-30 minutes'
        },
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 15m - КОРОТКОСТРОКОВИЙ ТРЕЙДИНГ
        # ═══════════════════════════════════════════════════════════════════════════
        '15m': {
            'rsi_period': 12,             # Коротко-середня
            'rsi_threshold_long': 71,     # Перекупленість
            'rsi_threshold_short': 29,    # Перепроданість
            'trailing_stop_percent': -0.008,  # -0.8%
            'min_divergence_candles': 2,  # 2 свічки для стабільності
            'stop_loss_percent': 1.0,     # 1.0% SL
            'take_profit_percent': 1.2,   # 1.2% TP
            'hma_fast_period': 9,         # Швидкий MA
            'hma_slow_period': 18,        # Повільний MA
            'expected_return': '+0.8% - +1.5%',
            'expected_max_loss': '-0.8%',
            'strategy_type': 'SHORT_TERM',
            'trades_per_day': '10-20',
            'hold_time': '15 min - 2 hours'
        },
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 1h - СЕРЕДНЬОСТРОКОВИЙ ТРЕЙДИНГ (СТАНДАРТ) ⭐
        # ═══════════════════════════════════════════════════════════════════════════
        '1h': {
            'rsi_period': 14,             # Стандартна довжина (ПОСЛ.)
            'rsi_threshold_long': 70,     # Стандартна перекупленість
            'rsi_threshold_short': 30,    # Стандартна перепроданість
            'trailing_stop_percent': -0.005,  # -0.5% (СТАНДАРТ)
            'min_divergence_candles': 2,  # 2 свічки (РЕКОМЕНДО)
            'stop_loss_percent': 0.8,     # 0.8% SL
            'take_profit_percent': 1.5,   # 1.5% TP
            'hma_fast_period': 9,         # Швидкий MA
            'hma_slow_period': 21,        # Повільний MA
            'expected_return': '+1.0% - +2.5%',
            'expected_max_loss': '-0.5%',
            'strategy_type': 'BALANCED',
            'trades_per_day': '3-8',
            'hold_time': '1-8 hours'
        },
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 4h - СЕРЕДНЬОСТРОКОВИЙ+ ТРЕЙДИНГ
        # ═══════════════════════════════════════════════════════════════════════════
        '4h': {
            'rsi_period': 16,             # Довша для більшої стабільності
            'rsi_threshold_long': 68,     # Трохи нижче (більш релаксований)
            'rsi_threshold_short': 32,    # Трохи вище
            'trailing_stop_percent': -0.004,  # -0.4% (менший для більших ходів)
            'min_divergence_candles': 2,  # 2 свічки
            'stop_loss_percent': 0.6,     # 0.6% SL
            'take_profit_percent': 2.0,   # 2.0% TP
            'hma_fast_period': 12,        # Швидкий MA
            'hma_slow_period': 24,        # Повільний MA
            'expected_return': '+1.5% - +3.5%',
            'expected_max_loss': '-0.4%',
            'strategy_type': 'SWING',
            'trades_per_day': '1-3',
            'hold_time': '4-24 hours'
        },
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 1d - ДОВГОСТРОКОВИЙ ТРЕЙДИНГ
        # ═══════════════════════════════════════════════════════════════════════════
        '1d': {
            'rsi_period': 21,             # Довга для дня (менш шуму)
            'rsi_threshold_long': 65,     # Більш релаксований поріг
            'rsi_threshold_short': 35,    # Більш релаксований
            'trailing_stop_percent': -0.003,  # -0.3% (маленький для великих ходів)
            'min_divergence_candles': 3,  # 3 свічки для потужного сигналу
            'stop_loss_percent': 0.5,     # 0.5% SL
            'take_profit_percent': 2.5,   # 2.5% TP
            'hma_fast_period': 14,        # Швидкий MA
            'hma_slow_period': 28,        # Повільний MA
            'expected_return': '+2.0% - +5.0%',
            'expected_max_loss': '-0.3%',
            'strategy_type': 'LONG_TERM',
            'trades_per_day': '1',
            'hold_time': '1-7 days'
        }
    }
    
    # Модифікатори для ПРОФІЛІВ (множать базові параметри)
    PROFILE_MULTIPLIERS = {
        'CONSERVATIVE': {
            'rsi_threshold_long': 1.08,     # +8% до порогу (жорсткіше)
            'rsi_threshold_short': 0.92,    # -8%
            'trailing_stop_percent': 0.60,  # 60% від базового (менше)
            'min_divergence_candles': 1.5,  # +50% свічок
            'stop_loss_percent': 1.2,       # 120% від базового (більше защиты)
            'take_profit_percent': 0.7,     # 70% від базового (менше)
        },
        'BALANCED': {
            'rsi_threshold_long': 1.0,      # Без змін (100%)
            'rsi_threshold_short': 1.0,
            'trailing_stop_percent': 1.0,   # Без змін
            'min_divergence_candles': 1.0,
            'stop_loss_percent': 1.0,
            'take_profit_percent': 1.0,
        },
        'AGGRESSIVE': {
            'rsi_threshold_long': 0.93,     # -7% до порогу (менш жорстко)
            'rsi_threshold_short': 1.07,    # +7%
            'trailing_stop_percent': 1.4,   # 140% від базового (більше)
            'min_divergence_candles': 0.7,  # -30% свічок (менш строго)
            'stop_loss_percent': 0.8,       # 80% від базового (менше защиты)
            'take_profit_percent': 1.4,     # 140% від базового (більше)
        },
        'SCALPING': {
            'rsi_threshold_long': 0.97,     # Трохи нижче порогу
            'rsi_threshold_short': 1.03,    # Трохи вище
            'trailing_stop_percent': 1.6,   # 160% від базового
            'min_divergence_candles': 0.5,  # -50% (мінімальні)
            'stop_loss_percent': 0.6,       # Менше защиты
            'take_profit_percent': 0.8,     # Менше профіта
        },
        'SWING': {
            'rsi_threshold_long': 1.05,     # Трохи вище порогу
            'rsi_threshold_short': 0.95,    # Трохи нижче
            'trailing_stop_percent': 0.8,   # 80% від базового (менше)
            'min_divergence_candles': 1.3,  # +30% свічок (більше строго)
            'stop_loss_percent': 0.9,       # 90% від базового
            'take_profit_percent': 1.2,     # 120% від базового
        }
    }
    
    @staticmethod
    def get_timeframe_config(timeframe='1h', profile='BALANCED'):
        """
        Отримати конфіг для конкретного таймфрейму та профілю
        
        Args:
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            profile: 'CONSERVATIVE', 'BALANCED', 'AGGRESSIVE', 'SCALPING', 'SWING'
        
        Returns:
            dict з параметрами
        """
        
        # Отримуємо базовий конфіг таймфрейму
        if timeframe not in TimeframeParameters.TIMEFRAME_CONFIGS:
            logger.warning(f"⚠️ Unknown timeframe: {timeframe}, using 1h")
            timeframe = '1h'
        
        base_config = TimeframeParameters.TIMEFRAME_CONFIGS[timeframe].copy()
        
        # Отримуємо модифікатор профілю
        if profile not in TimeframeParameters.PROFILE_MULTIPLIERS:
            logger.warning(f"⚠️ Unknown profile: {profile}, using BALANCED")
            profile = 'BALANCED'
        
        multipliers = TimeframeParameters.PROFILE_MULTIPLIERS[profile]
        
        # Застосовуємо модифікатори
        config = base_config.copy()
        
        # Для RSI thresholds
        config['rsi_threshold_long'] = int(
            base_config['rsi_threshold_long'] * multipliers['rsi_threshold_long']
        )
        config['rsi_threshold_short'] = int(
            base_config['rsi_threshold_long'] - (base_config['rsi_threshold_long'] - base_config['rsi_threshold_short']) * multipliers['rsi_threshold_short']
        )
        
        # Для числових параметрів
        config['trailing_stop_percent'] = round(
            base_config['trailing_stop_percent'] * multipliers['trailing_stop_percent'], 4
        )
        config['min_divergence_candles'] = max(1, int(
            base_config['min_divergence_candles'] * multipliers['min_divergence_candles']
        ))
        config['stop_loss_percent'] = round(
            base_config['stop_loss_percent'] * multipliers['stop_loss_percent'], 2
        )
        config['take_profit_percent'] = round(
            base_config['take_profit_percent'] * multipliers['take_profit_percent'], 2
        )
        
        # Додаємо інформацію про профіль та таймфрейм
        config['profile'] = profile
        config['timeframe'] = timeframe
        
        return config
    
    @staticmethod
    def get_all_timeframes():
        """Отримати список всіх таймфреймів"""
        return list(TimeframeParameters.TIMEFRAME_CONFIGS.keys())
    
    @staticmethod
    def get_all_profiles():
        """Отримати список всіх профілів"""
        return list(TimeframeParameters.PROFILE_MULTIPLIERS.keys())
    
    @staticmethod
    def print_config_table(timeframe='1h', profile='BALANCED'):
        """Вивести конфіг красиво"""
        config = TimeframeParameters.get_timeframe_config(timeframe, profile)
        
        print("\n" + "=" * 70)
        print(f"📊 CONFIG: {profile} | {timeframe.upper()}")
        print("=" * 70)
        
        print(f"RSI Period:               {config['rsi_period']}")
        print(f"RSI Long Threshold:       {config['rsi_threshold_long']}")
        print(f"RSI Short Threshold:      {config['rsi_threshold_short']}")
        print(f"Trailing Stop:            {config['trailing_stop_percent']*100:.2f}%")
        print(f"Min Divergence Candles:   {config['min_divergence_candles']}")
        print(f"Stop Loss:                {config['stop_loss_percent']:.2f}%")
        print(f"Take Profit:              {config['take_profit_percent']:.2f}%")
        print(f"HMA Fast Period:          {config['hma_fast_period']}")
        print(f"HMA Slow Period:          {config['hma_slow_period']}")
        print(f"Expected Return:          {config['expected_return']}")
        print(f"Expected Max Loss:        {config['expected_max_loss']}")
        print(f"Strategy Type:            {config['strategy_type']}")
        print(f"Trades Per Day:           {config['trades_per_day']}")
        print(f"Hold Time:                {config['hold_time']}")
        print("=" * 70 + "\n")
    
    @staticmethod
    def print_all_timeframes(profile='BALANCED'):
        """Вивести всі таймфрейми для профілю"""
        print("\n" + "🎯" * 35)
        print(f"ВСІХ ТАЙМФРЕЙМИ: {profile}")
        print("🎯" * 35 + "\n")
        
        timeframes = TimeframeParameters.get_all_timeframes()
        
        for tf in timeframes:
            TimeframeParameters.print_config_table(tf, profile)
    
    @staticmethod
    def print_all_profiles(timeframe='1h'):
        """Вивести всі профілі для таймфрейму"""
        print("\n" + "🎯" * 35)
        print(f"ВСІ ПРОФІЛІ: {timeframe.upper()}")
        print("🎯" * 35 + "\n")
        
        profiles = TimeframeParameters.get_all_profiles()
        
        for profile in profiles:
            TimeframeParameters.print_config_table(timeframe, profile)


# ════════════════════════════════════════════════════════════════════════════
# ПРИКЛАДИ ВИКОРИСТАННЯ
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ПРИКЛАД 1: Отримати конфіг
    print("\n✅ ПРИКЛАД 1: Отримати конфіг")
    config = TimeframeParameters.get_timeframe_config('1h', 'BALANCED')
    print(f"RSI Threshold: {config['rsi_threshold_long']}")
    print(f"Trailing Stop: {config['trailing_stop_percent']*100}%")
    
    # ПРИКЛАД 2: Вивести таблицю
    print("\n✅ ПРИКЛАД 2: Вивести конфіг")
    TimeframeParameters.print_config_table('1h', 'BALANCED')
    
    # ПРИКЛАД 3: Всі таймфрейми для 1 профілю
    print("\n✅ ПРИКЛАД 3: Всі таймфрейми")
    TimeframeParameters.print_all_timeframes('BALANCED')
    
    # ПРИКЛАД 4: Всі профілі для 1 таймфрейму
    print("\n✅ ПРИКЛАД 4: Всі профілі")
    TimeframeParameters.print_all_profiles('1h')
