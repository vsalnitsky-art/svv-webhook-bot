#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from timeframe_parameters import TimeframeParameters

logger = logging.getLogger(__name__)

class SettingsManager:
    """
    Управління налаштуваннями з автоматичним підлаштуванням під таймфрейм
    
    Коли змінюється таймфрейм → всі параметри автоматично оновлюються!
    """
    
    DEFAULT_SETTINGS = {
        # ОСНОВНІ ПАРАМЕТРИ
        'leverage': 10,
        'riskPercent': 2,
        'fixedSL': 1.0,
        
        # ТАЙМФРЕЙМ (ключовий параметр!)
        'timeframe': '1h',              # ⭐ ОСНОВНИЙ ПАРАМЕТР
        'profile': 'BALANCED',          # ⭐ ПРОФІЛЬ
        
        # Smart Exit налаштування (будуть автоматично встановлені)
        'rsi_period': 14,
        'rsi_threshold': 70,
        'trailing_stop_percent': -0.005,
        'min_divergence_candles': 2,
        
        # TP/SL стратегія
        'tp_mode': 'Fixed_1_50',        # Fixed_1_50 або Ladder_3
        'takeProfitPercent': 1.5,
        
        # Інші параметри
        'enableSmartExit': True,
        'enableAutoSync': True,
    }
    
    def __init__(self, settings_file='settings.json'):
        self.settings_file = settings_file
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load_settings()
    
    def load_settings(self):
        """Завантажити налаштування з файлу"""
        try:
            with open(self.settings_file, 'r') as f:
                loaded = json.load(f)
                self.settings.update(loaded)
                logger.info(f"✅ Settings loaded from {self.settings_file}")
        except FileNotFoundError:
            logger.info(f"⚠️ Settings file not found, using defaults")
            self.save_settings()
        except Exception as e:
            logger.error(f"❌ Error loading settings: {e}")
    
    def get_all(self):
        """Отримати всі налаштування"""
        return self.settings.copy()
    
    def get(self, key, default=None):
        """Отримати конкретне налаштування"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Встановити налаштування"""
        self.settings[key] = value
        logger.info(f"✅ Setting {key} = {value}")
    
    def update_for_timeframe(self, timeframe=None, profile=None):
        """
        ✅ ГОЛОВНА ЛОГІКА: Оновити всі параметри для таймфрейму
        
        Коли користувач змінює таймфрейм або профіль,
        всі параметри автоматично перелаштовуються!
        
        Args:
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            profile: 'CONSERVATIVE', 'BALANCED', 'AGGRESSIVE', 'SCALPING', 'SWING'
        """
        
        # Використовуємо поточні значення якщо не передано
        timeframe = timeframe or self.settings.get('timeframe', '1h')
        profile = profile or self.settings.get('profile', 'BALANCED')
        
        # Отримуємо конфіг для таймфрейму та профілю
        config = TimeframeParameters.get_timeframe_config(timeframe, profile)
        
        logger.info(f"🎯 Updating parameters for {profile} | {timeframe.upper()}")
        
        # Оновлюємо основні параметри
        self.settings['timeframe'] = timeframe
        self.settings['profile'] = profile
        
        # Оновлюємо Smart Exit параметри
        self.settings['rsi_period'] = config['rsi_period']
        self.settings['rsi_threshold'] = config['rsi_threshold_long']  # Для LONG
        self.settings['rsi_threshold_short'] = config['rsi_threshold_short']  # Для SHORT
        self.settings['trailing_stop_percent'] = config['trailing_stop_percent']
        self.settings['min_divergence_candles'] = config['min_divergence_candles']
        self.settings['fixedSL'] = config['stop_loss_percent']
        self.settings['takeProfitPercent'] = config['take_profit_percent']
        
        # Додаткові параметри
        self.settings['hma_fast_period'] = config['hma_fast_period']
        self.settings['hma_slow_period'] = config['hma_slow_period']
        self.settings['expected_return'] = config['expected_return']
        self.settings['expected_max_loss'] = config['expected_max_loss']
        self.settings['strategy_type'] = config['strategy_type']
        self.settings['hold_time'] = config['hold_time']
        
        logger.info(
            f"✅ Parameters updated:\n"
            f"   RSI Period: {config['rsi_period']}\n"
            f"   RSI Threshold: {config['rsi_threshold_long']}\n"
            f"   Trailing Stop: {config['trailing_stop_percent']*100:.2f}%\n"
            f"   SL: {config['stop_loss_percent']:.2f}%\n"
            f"   TP: {config['take_profit_percent']:.2f}%\n"
            f"   Expected: {config['expected_return']}"
        )
        
        # Зберігаємо
        self.save_settings()
        
        return config
    
    def save_settings(self):
        """Зберегти налаштування в файл"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            logger.info(f"✅ Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"❌ Error saving settings: {e}")
    
    def get_timeframe_config(self):
        """Отримати повний конфіг поточного таймфрейму та профілю"""
        timeframe = self.settings.get('timeframe', '1h')
        profile = self.settings.get('profile', 'BALANCED')
        return TimeframeParameters.get_timeframe_config(timeframe, profile)
    
    def print_current_config(self):
        """Вивести поточні налаштування"""
        timeframe = self.settings.get('timeframe', '1h')
        profile = self.settings.get('profile', 'BALANCED')
        TimeframeParameters.print_config_table(timeframe, profile)
    
    def get_available_timeframes(self):
        """Отримати список доступних таймфреймів"""
        return TimeframeParameters.get_all_timeframes()
    
    def get_available_profiles(self):
        """Отримати список доступних профілів"""
        return TimeframeParameters.get_all_profiles()
    
    def validate_timeframe(self, timeframe):
        """Перевірити, чи таймфрейм валідний"""
        return timeframe in self.get_available_timeframes()
    
    def validate_profile(self, profile):
        """Перевірити, чи профіль валідний"""
        return profile in self.get_available_profiles()


# ════════════════════════════════════════════════════════════════════════════
# ПРИКЛАДИ ВИКОРИСТАННЯ
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ІНІЦІАЛІЗАЦІЯ
    print("\n✅ ІНІЦІАЛІЗАЦІЯ")
    settings = SettingsManager()
    
    # ПРИКЛАД 1: Отримати всі налаштування
    print("\n✅ ПРИКЛАД 1: Поточні налаштування")
    all_settings = settings.get_all()
    print(f"Timeframe: {all_settings['timeframe']}")
    print(f"Profile: {all_settings['profile']}")
    print(f"RSI Threshold: {all_settings['rsi_threshold']}")
    
    # ПРИКЛАД 2: Змінити таймфрейм на 5m
    print("\n✅ ПРИКЛАД 2: Змінити на 5m SCALPING")
    settings.update_for_timeframe('5m', 'SCALPING')
    settings.print_current_config()
    
    # ПРИКЛАД 3: Змінити на 4h AGGRESSIVE
    print("\n✅ ПРИКЛАД 3: Змінити на 4h AGGRESSIVE")
    settings.update_for_timeframe('4h', 'AGGRESSIVE')
    settings.print_current_config()
    
    # ПРИКЛАД 4: Змінити на 1d CONSERVATIVE
    print("\n✅ ПРИКЛАД 4: Змінити на 1d CONSERVATIVE")
    settings.update_for_timeframe('1d', 'CONSERVATIVE')
    settings.print_current_config()
    
    # ПРИКЛАД 5: Отримати доступні таймфрейми та профілі
    print("\n✅ ПРИКЛАД 5: Доступні опції")
    print(f"Timeframes: {settings.get_available_timeframes()}")
    print(f"Profiles: {settings.get_available_profiles()}")
