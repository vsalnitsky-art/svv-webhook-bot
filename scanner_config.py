"""
Scanner Configuration v2.2
==========================
НОВІ МОЖЛИВОСТІ:
- Розділені параметри RSI для входу та виходу
- Окремі параметри для сканера ринку
- Повна українізація коментарів
"""

from typing import Dict, Any, List
import json
import os

class ScannerConfig:
    """
    Конфігурація сканера з розділеними параметрами
    """
    
    def __init__(self):
        # Базові параметри
        self.trading_style = 'daytrading'
        self.aggressiveness = 'auto'
        self.automation_mode = 'semi_auto'
        self.indicator_timeframe = '240'
        
        # ⭐ НОВЕ: Параметри входу (ENTRY)
        self.entry_params = self._get_default_entry_params()
        
        # ⭐ НОВЕ: Параметри виходу (EXIT)
        self.exit_params = self._get_default_exit_params()
        
        # ⭐ НОВЕ: Параметри сканера (SCANNER)
        self.scanner_params = self._get_default_scanner_params()
        
        # Старі параметри (для сумісності)
        self.indicator_params = {}
        self.risk_params = self._get_default_risk_params()
        self.auto_close_params = self._get_default_auto_close_params()
        self.auto_open_enabled = True
        
        # Keep-Alive
        self.keep_alive_enabled = False
        self.keep_alive_interval = 300
        
        # Пресети
        self.TRADING_STYLES = {
            'scalping': {
                'timeframe': '15',
                'entry_rsi_oversold': 35,
                'entry_rsi_overbought': 65,
                'exit_rsi_oversold': 40,
                'exit_rsi_overbought': 60,
            },
            'daytrading': {
                'timeframe': '240',
                'entry_rsi_oversold': 45,
                'entry_rsi_overbought': 55,
                'exit_rsi_oversold': 50,
                'exit_rsi_overbought': 50,
            },
            'swing': {
                'timeframe': '1D',
                'entry_rsi_oversold': 30,
                'entry_rsi_overbought': 70,
                'exit_rsi_oversold': 35,
                'exit_rsi_overbought': 65,
            }
        }
        
        self.AGGRESSIVENESS_PRESETS = {
            'conservative': {
                'entry_require_volume': True,
                'entry_trend_confirmation': True,
                'exit_require_volume': True,
                'min_peak_strength': 3,
            },
            'auto': {
                'entry_require_volume': False,
                'entry_trend_confirmation': False,
                'exit_require_volume': False,
                'min_peak_strength': 1,
            },
            'aggressive': {
                'entry_require_volume': False,
                'entry_trend_confirmation': False,
                'exit_require_volume': False,
                'min_peak_strength': 1,
            }
        }
        
        self.AUTOMATION_PRESETS = {
            'full_auto': {
                'auto_open_enabled': True,
                'auto_close_enabled': True,
                'log_decisions': True,
            },
            'semi_auto': {
                'auto_open_enabled': True,
                'auto_close_enabled': False,
                'log_decisions': True,
            },
            'manual': {
                'auto_open_enabled': False,
                'auto_close_enabled': False,
                'log_decisions': False,
            }
        }
    
    def _get_default_entry_params(self) -> Dict[str, Any]:
        """
        Параметри для ВХОДУ в позицію
        """
        return {
            # Таймфрейм для входу
            'timeframe': '240',  # 4 години
            
            # RSI зони для входу
            'rsi_oversold': 45,      # Long при RSI < 45
            'rsi_overbought': 55,    # Short при RSI > 55
            'rsi_period': 14,        # Період RSI
            
            # MFI для входу
            'mfi_enabled': True,     # Використовувати MFI
            'mfi_period': 20,
            'mfi_fast_ema': 5,
            'mfi_slow_ema': 13,
            
            # Фільтри для входу
            'require_volume': False,         # Вимагати об'ємного підтвердження
            'trend_confirmation': False,     # Вимагати підтвердження тренду
            'min_peak_strength': 1,          # Мінімальна сила піку
        }
    
    def _get_default_exit_params(self) -> Dict[str, Any]:
        """
        Параметри для ВИХОДУ з позиції
        """
        return {
            # Таймфрейм для виходу
            'timeframe': '240',  # 4 години (може бути інший!)
            
            # RSI зони для виходу
            'rsi_oversold': 50,      # Закрити Long при RSI > 50
            'rsi_overbought': 50,    # Закрити Short при RSI < 50
            'rsi_period': 14,
            
            # MFI для виходу
            'mfi_enabled': True,
            'mfi_period': 20,
            
            # Фільтри для виходу
            'require_volume': False,
            'use_strong_signals_only': False,  # Тільки сильні сигнали
        }
    
    def _get_default_scanner_params(self) -> Dict[str, Any]:
        """
        Окремі параметри для СКАНЕРА ринку
        (не впливають на торгівлю!)
        """
        return {
            # Таймфрейм для сканування
            'timeframe': '240',  # 4 години
            
            # RSI зони для пошуку кандидатів
            'rsi_oversold': 45,      # Шукати Long при RSI < 45
            'rsi_overbought': 55,    # Шукати Short при RSI > 55
            'rsi_period': 14,
            
            # Фільтри для сканера
            'require_volume': False,
            'trend_confirmation': False,
            'min_peak_strength': 1,
            
            # Об'ємні фільтри
            'min_volume_24h': 3_000_000,     # $3M
            'min_price_change_24h': 0.8,     # 0.8%
            'min_market_cap': 50_000_000,    # $50M
            
            # Налаштування сканування
            'enabled': False,
            'scan_interval': 60,
            'batch_size': 30,
            'parallel_processing': False,
            'top_candidates_count': 10,
            'min_signal_strength': 'regular',
            'show_direction': 'both',
        }
    
    def _get_default_risk_params(self) -> Dict[str, Any]:
        """Параметри управління ризиками"""
        return {
            'max_positions': 3,
            'position_size_percent': 10.0,
            'daily_loss_limit_percent': -5.0,  # Негативне!
            'default_leverage': 20,
            'min_balance_reserve': 100.0,
            'check_correlation': True,
            'max_same_direction': 2,
            'blacklist_symbols': ['LUNAUSDT', 'USTUSDT', 'TITANUSDT'],
        }
    
    def _get_default_auto_close_params(self) -> Dict[str, Any]:
        """Параметри автозакриття"""
        return {
            'enabled': False,
            'use_strong_signals': True,
            'use_regular_signals': False,
            'extreme_rsi_close': True,
            'use_obv_confirmation': True,
            'obv_ema_period': 20,
            'obv_trend_candles': 3,
            'obv_sensitivity': 'high',
            'log_all_decisions': True,
        }
    
    def get_entry_params(self) -> Dict[str, Any]:
        """Отримати параметри входу"""
        return self.entry_params.copy()
    
    def get_exit_params(self) -> Dict[str, Any]:
        """Отримати параметри виходу"""
        return self.exit_params.copy()
    
    def get_scanner_params(self) -> Dict[str, Any]:
        """Отримати параметри сканера"""
        return self.scanner_params.copy()
    
    def get_indicator_params(self) -> Dict[str, Any]:
        """
        Отримати параметри для індикатора (для торгівлі)
        Використовує ENTRY параметри за замовчуванням
        """
        return {
            'timeframe': self.entry_params['timeframe'],
            'rsi_length': self.entry_params['rsi_period'],
            'oversold': self.entry_params['rsi_oversold'],
            'overbought': self.entry_params['rsi_overbought'],
            'mfi_length': self.entry_params['mfi_period'],
            'fast_mfi_ema': self.entry_params['mfi_fast_ema'],
            'slow_mfi_ema': self.entry_params['mfi_slow_ema'],
            'require_volume': self.entry_params['require_volume'],
            'trend_confirmation': self.entry_params['trend_confirmation'],
            'min_peak_strength': self.entry_params['min_peak_strength'],
            'show_signals': True,
            'show_bullish_signals': True,
            'show_bearish_signals': True,
            'bot_risk': 5.0,
            'bot_leverage': 20,
            'bot_tp': 0.5,
            'bot_sl': 0.5,
            'enable_alerts': False,
        }
    
    def update_entry_param(self, key: str, value: Any):
        """Оновити параметр входу"""
        if key in self.entry_params:
            self.entry_params[key] = value
    
    def update_exit_param(self, key: str, value: Any):
        """Оновити параметр виходу"""
        if key in self.exit_params:
            self.exit_params[key] = value
    
    def update_scanner_param(self, key: str, value: Any):
        """Оновити параметр сканера"""
        if key in self.scanner_params:
            self.scanner_params[key] = value
    
    def update_param(self, category: str, key: str, value: Any):
        """Оновити параметр (стара сумісність)"""
        if category == 'risk':
            if key in self.risk_params:
                self.risk_params[key] = value
        elif category == 'indicator':
            # Оновлюємо entry параметри
            mapping = {
                'rsi_period': 'rsi_period',
                'rsi_oversold': 'rsi_oversold',
                'rsi_overbought': 'rsi_overbought',
                'mfi_period': 'mfi_period',
                'mfi_fast_ema': 'mfi_fast_ema',
                'mfi_slow_ema': 'mfi_slow_ema',
            }
            if key in mapping:
                self.entry_params[mapping[key]] = value
    
    def set_timeframe(self, timeframe: str):
        """Встановити таймфрейм для всіх"""
        self.indicator_timeframe = timeframe
        self.entry_params['timeframe'] = timeframe
        self.exit_params['timeframe'] = timeframe
        # Сканер має свій окремий таймфрейм!
    
    def update_trading_style(self, style: str):
        """Оновити стиль торгівлі"""
        if style in self.TRADING_STYLES:
            self.trading_style = style
            preset = self.TRADING_STYLES[style]
            
            # Застосувати до entry
            self.entry_params['timeframe'] = preset['timeframe']
            self.entry_params['rsi_oversold'] = preset['entry_rsi_oversold']
            self.entry_params['rsi_overbought'] = preset['entry_rsi_overbought']
            
            # Застосувати до exit
            self.exit_params['timeframe'] = preset['timeframe']
            self.exit_params['rsi_oversold'] = preset['exit_rsi_oversold']
            self.exit_params['rsi_overbought'] = preset['exit_rsi_overbought']
    
    def update_aggressiveness(self, level: str):
        """Оновити агресивність"""
        if level in self.AGGRESSIVENESS_PRESETS:
            self.aggressiveness = level
            preset = self.AGGRESSIVENESS_PRESETS[level]
            
            # Застосувати до entry
            self.entry_params['require_volume'] = preset['entry_require_volume']
            self.entry_params['trend_confirmation'] = preset['entry_trend_confirmation']
            self.entry_params['min_peak_strength'] = preset['min_peak_strength']
            
            # Застосувати до exit
            self.exit_params['require_volume'] = preset['exit_require_volume']
    
    def update_automation(self, mode: str):
        """Оновити режим автоматизації"""
        if mode in self.AUTOMATION_PRESETS:
            self.automation_mode = mode
            preset = self.AUTOMATION_PRESETS[mode]
            
            self.auto_open_enabled = preset['auto_open_enabled']
            self.auto_close_params['enabled'] = preset['auto_close_enabled']
            self.auto_close_params['log_all_decisions'] = preset['log_decisions']

# Глобальний екземпляр
config = ScannerConfig()
