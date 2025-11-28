"""
Scanner Configuration Manager
Управление всеми параметрами сканера с пресетами
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ScannerConfig:
    """Класс управления конфигурацией сканера"""
    
    # ==================== ПРЕСЕТЫ СТИЛЕЙ ТОРГОВЛИ ====================
    TRADING_STYLE_PRESETS = {
        'scalping': {
            'name': 'Скальпинг',
            'description': 'Быстрые сделки, высокая частота',
            'rsi_length': 10,
            'oversold': 35,
            'overbought': 65,
            'min_peak_strength': 1,
            'scan_interval': 30,
            'min_hold_time': 60,  # 1 минута
        },
        'daytrading': {
            'name': 'Дейтрейдинг',
            'description': 'Торговля внутри дня (15м - 4ч)',
            'rsi_length': 14,
            'oversold': 30,
            'overbought': 70,
            'min_peak_strength': 2,
            'scan_interval': 60,
            'min_hold_time': 300,  # 5 минут
        },
        'swing': {
            'name': 'Свинг',
            'description': 'Позиционная торговля (4ч - 1д)',
            'rsi_length': 21,
            'oversold': 25,
            'overbought': 75,
            'min_peak_strength': 3,
            'scan_interval': 300,
            'min_hold_time': 1800,  # 30 минут
        }
    }
    
    # ==================== ПРЕСЕТЫ АГРЕССИВНОСТИ ====================
    AGGRESSIVENESS_PRESETS = {
        'auto': {
            'name': 'Автоматическая подстройка',
            'enabled': True,
            'volatility_threshold': 5.0,
            'high_volatility': {
                'oversold': 25,
                'overbought': 75,
                'require_volume': True,
                'trend_confirmation': True,
            },
            'low_volatility': {
                'oversold': 35,
                'overbought': 65,
                'require_volume': False,
                'trend_confirmation': False,
            }
        },
        'conservative': {
            'name': 'Консервативный',
            'enabled': False,
            'oversold': 25,
            'overbought': 75,
            'require_volume': True,
            'trend_confirmation': True,
            'min_peak_strength': 3,
        },
        'aggressive': {
            'name': 'Агрессивный',
            'enabled': False,
            'oversold': 35,
            'overbought': 65,
            'require_volume': False,
            'trend_confirmation': False,
            'min_peak_strength': 1,
        }
    }
    
    # ==================== ПРЕСЕТЫ АВТОМАТИЗАЦИИ ====================
    AUTOMATION_PRESETS = {
        'semi_auto': {
            'name': 'Полуавтоматический',
            'auto_close_enabled': True,
            'auto_open_enabled': False,
            'require_confirmation': True,
            'show_notifications': True,
            'log_decisions': True,
        },
        'full_auto': {
            'name': 'Полностью автоматический',
            'auto_close_enabled': True,
            'auto_open_enabled': True,
            'require_confirmation': False,
            'show_notifications': True,
            'log_decisions': True,
        }
    }
    
    def __init__(self):
        """Инициализация с параметрами по умолчанию"""
        # Текущие выборы
        self.trading_style = 'daytrading'  # По умолчанию
        self.aggressiveness = 'auto'       # По умолчанию
        self.automation_mode = 'semi_auto' # По умолчанию
        
        # Таймфрейм индикатора (по умолчанию 4 часа)
        self.indicator_timeframe = '240'  # 240 минут = 4 часа
        
        # Keep-Alive (Self-Ping) ⭐
        self.keep_alive_enabled = True  # По умолчанию ВКЛЮЧЕНО
        self.keep_alive_interval = 300  # 5 минут (в секундах)
        
        # Параметры индикатора (будут переопределены при загрузке пресета)
        self.indicator_params = {}
        
        # Параметры Risk Management
        self.risk_params = self._get_default_risk_params()
        
        # Параметры автозакрытия
        self.auto_close_params = self._get_default_auto_close_params()
        
        # Параметры сканирования
        self.scanner_params = self._get_default_scanner_params()
        
        # Загрузить параметры из выбранных пресетов
        self._apply_presets()
        
        logger.info("✅ Scanner config initialized with defaults")
    
    def _apply_presets(self):
        """Применить выбранные пресеты"""
        # Применить стиль торговли
        style = self.TRADING_STYLE_PRESETS[self.trading_style]
        self.indicator_params.update({
            'rsi_length': style['rsi_length'],
            'oversold': style['oversold'],
            'overbought': style['overbought'],
            'min_peak_strength': style['min_peak_strength'],
        })
        self.scanner_params['scan_interval'] = style['scan_interval']
        self.auto_close_params['min_hold_time'] = style['min_hold_time']
        
        # Применить агрессивность
        agg = self.AGGRESSIVENESS_PRESETS[self.aggressiveness]
        if self.aggressiveness == 'auto':
            self.aggressiveness_config = agg
        else:
            self.indicator_params.update({
                'oversold': agg['oversold'],
                'overbought': agg['overbought'],
                'require_volume': agg['require_volume'],
                'trend_confirmation': agg['trend_confirmation'],
                'min_peak_strength': agg.get('min_peak_strength', 2),
            })
        
        # Применить автоматизацию
        auto = self.AUTOMATION_PRESETS[self.automation_mode]
        self.auto_close_params.update({
            'enabled': auto['auto_close_enabled'],
            'log_all_decisions': auto['log_decisions'],
        })
        self.auto_open_enabled = auto['auto_open_enabled']
    
    def _get_default_risk_params(self) -> Dict[str, Any]:
        """Параметры Risk Management по умолчанию"""
        return {
            'max_positions': 3,
            'max_position_size_percent': 10.0,
            'daily_loss_limit_percent': 5.0,
            'max_leverage': 20,
            'min_balance_reserve': 100.0,
            'check_correlation': True,
            'max_same_direction': 2,
            'blacklist_symbols': ['LUNAUSDT', 'USTUSDT', 'TITANUSDT'],
        }
    
    def _get_default_auto_close_params(self) -> Dict[str, Any]:
        """Параметры автозакрытия по умолчанию"""
        return {
            'enabled': False,  # ⭐ По умолчанию ВЫКЛЮЧЕНО
            'use_strong_signals': True,
            'use_regular_signals': False,
            'extreme_rsi_close': True,
            
            # OBV Configuration (НОВОЕ) ⭐
            'use_obv_confirmation': True,  # По умолчанию ВКЛЮЧЕНО
            'obv_ema_period': 20,
            'obv_trend_candles': 3,  # Для medium sensitivity
            'obv_sensitivity': 'high',  # low/medium/high (default: high = 2 свечи)
            'rsi_exit_mode': 'wait_zone_exit',  # Если OBV выключен: 'immediate' или 'wait_zone_exit'
            
            # MFI Configuration (обновлено)
            'confirm_with_mfi': False,  # По умолчанию ВЫКЛЮЧЕНО ⭐
            'mfi_check_mode': 'after_obv',  # 'before_obv', 'after_obv', 'with_obv'
            'confirm_with_cloud': True,
            
            'min_hold_time': 300,  # 5 минут
            'ignore_small_profit_percent': 0.5,
            'consider_unrealized_pnl': True,
            'log_all_decisions': True,
            'save_close_analytics': True,
        }
    
    def _get_default_scanner_params(self) -> Dict[str, Any]:
        """Параметры сканирования по умолчанию"""
        return {
            'enabled': False,  # ⭐ По умолчанию ВЫКЛЮЧЕНО
            'scan_interval': 60,
            'batch_size': 30,              # ✅ Зменшено з 50 до 30
            'min_volume_24h': 3_000_000,    # $3M ✅ (збільшено з $1M для швидкості)
            'min_price_change_24h': 0.8,    # 0.8% ✅ (збільшено з 0.5% для швидкості)
            'max_spread_percent': 0.5,
            'min_market_cap': 50_000_000,   # $50M ✅ (було $100M)
            'top_candidates_count': 10,
            'show_direction': 'both',
            'min_signal_strength': 'regular',  # ✅ Залишаємо regular (не strong!)
            'use_cache': True,
            'cache_ttl': 30,
            'parallel_processing': False,   # ✅ ВИМКНЕНО для стабільності на Render
            'min_rsi_for_long': 35,         # ✅ НОВИЙ параметр (ширша зона)
            'max_rsi_for_short': 65,        # ✅ НОВИЙ параметр (ширша зона)
        }
    
    def get_indicator_params(self) -> Dict[str, Any]:
        """
        Получить параметры для RSIMFIIndicator
        """
        base_params = {
            # Timeframe (ВАЖНО!)
            'timeframe': self.indicator_timeframe,
            
            # RSI
            'rsi_length': self.indicator_params.get('rsi_length', 14),
            'oversold': self.indicator_params.get('oversold', 30),
            'overbought': self.indicator_params.get('overbought', 70),
            
            # MFI
            'mfi_length': 20,
            'fast_mfi_ema': 5,
            'slow_mfi_ema': 13,
            'cloud_opacity': 40,
            
            # Фильтры
            'require_volume': self.indicator_params.get('require_volume', True),
            'trend_confirmation': self.indicator_params.get('trend_confirmation', True),
            'min_peak_strength': self.indicator_params.get('min_peak_strength', 2),
            
            # Визуализация
            'show_signals': True,
            'show_bullish_signals': True,
            'show_bearish_signals': True,
            
            # Бот (не используются в сканере, но нужны для индикатора)
            'bot_risk': 5.0,
            'bot_leverage': 20,
            'bot_tp': 0.5,
            'bot_sl': 0.5,
            
            # Алерты
            'enable_alerts': False,  # Отключено для сканера
        }
        
        return base_params
    
    def get_risk_params(self) -> Dict[str, Any]:
        """Получить параметры Risk Management"""
        return self.risk_params.copy()
    
    def get_auto_close_params(self) -> Dict[str, Any]:
        """Получить параметры автозакрытия"""
        return self.auto_close_params.copy()
    
    def get_scanner_params(self) -> Dict[str, Any]:
        """Получить параметры сканирования"""
        return self.scanner_params.copy()
    
    def update_trading_style(self, style: str):
        """Обновить стиль торговли"""
        if style in self.TRADING_STYLE_PRESETS:
            self.trading_style = style
            self._apply_presets()
            logger.info(f"✅ Trading style updated: {style}")
        else:
            logger.warning(f"⚠️ Unknown trading style: {style}")
    
    def update_aggressiveness(self, mode: str):
        """Обновить агрессивность"""
        if mode in self.AGGRESSIVENESS_PRESETS:
            self.aggressiveness = mode
            self._apply_presets()
            logger.info(f"✅ Aggressiveness updated: {mode}")
        else:
            logger.warning(f"⚠️ Unknown aggressiveness mode: {mode}")
    
    def update_automation(self, mode: str):
        """Обновить режим автоматизации"""
        if mode in self.AUTOMATION_PRESETS:
            self.automation_mode = mode
            self._apply_presets()
            logger.info(f"✅ Automation mode updated: {mode}")
        else:
            logger.warning(f"⚠️ Unknown automation mode: {mode}")
    
    def update_param(self, category: str, key: str, value: Any):
        """
        Обновить конкретный параметр
        
        Args:
            category: 'indicator', 'risk', 'auto_close', 'scanner'
            key: Название параметра
            value: Новое значение
        """
        if category == 'indicator':
            self.indicator_params[key] = value
        elif category == 'risk':
            self.risk_params[key] = value
        elif category == 'auto_close':
            self.auto_close_params[key] = value
        elif category == 'scanner':
            self.scanner_params[key] = value
        else:
            logger.warning(f"⚠️ Unknown category: {category}")
            return
        
        logger.info(f"✅ Parameter updated: {category}.{key} = {value}")
    
    def get_current_volatility(self, market_data: Dict) -> float:
        """
        Рассчитать текущую волатильность рынка
        (Упрощённая версия, можно улучшить)
        """
        try:
            # Пример: берём среднее изменение цены по топ-монетам
            changes = [abs(float(coin.get('price24hPcnt', 0))) 
                      for coin in market_data[:10]]
            avg_volatility = sum(changes) / len(changes) if changes else 0
            return avg_volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def apply_auto_aggressiveness(self, current_volatility: float):
        """
        Применить автоматическую подстройку агрессивности
        на основе волатильности рынка
        """
        if self.aggressiveness != 'auto':
            return
        
        config = self.aggressiveness_config
        threshold = config['volatility_threshold']
        
        if current_volatility > threshold:
            # Высокая волатильность → консервативно
            params = config['high_volatility']
            mode = 'консервативные'
        else:
            # Низкая волатильность → агрессивно
            params = config['low_volatility']
            mode = 'агрессивные'
        
        # Применить параметры
        self.indicator_params.update(params)
        logger.info(f"🎚️ Auto-adjusted to {mode} (volatility: {current_volatility:.2f}%)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Экспорт конфигурации в словарь"""
        return {
            'trading_style': self.trading_style,
            'aggressiveness': self.aggressiveness,
            'automation_mode': self.automation_mode,
            'indicator_timeframe': self.indicator_timeframe,
            'keep_alive_enabled': self.keep_alive_enabled,  # ⭐ НОВОЕ
            'keep_alive_interval': self.keep_alive_interval,  # ⭐ НОВОЕ
            'indicator_params': self.indicator_params,
            'risk_params': self.risk_params,
            'auto_close_params': self.auto_close_params,
            'scanner_params': self.scanner_params,
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Импорт конфигурации из словаря"""
        self.trading_style = data.get('trading_style', 'daytrading')
        self.aggressiveness = data.get('aggressiveness', 'auto')
        self.automation_mode = data.get('automation_mode', 'semi_auto')
        self.indicator_timeframe = data.get('indicator_timeframe', '240')
        self.keep_alive_enabled = data.get('keep_alive_enabled', True)  # ⭐ НОВОЕ
        self.keep_alive_interval = data.get('keep_alive_interval', 300)  # ⭐ НОВОЕ
        self.indicator_params = data.get('indicator_params', {})
        self.risk_params = data.get('risk_params', self._get_default_risk_params())
        self.auto_close_params = data.get('auto_close_params', self._get_default_auto_close_params())
        self.scanner_params = data.get('scanner_params', self._get_default_scanner_params())
        logger.info("✅ Config loaded from dict")
    
    def save_to_json(self, filepath: str = 'scanner_config.json'):
        """Сохранить конфигурацию в JSON файл"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Config saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def load_from_json(self, filepath: str = 'scanner_config.json'):
        """Загрузить конфигурацию из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.from_dict(data)
            logger.info(f"✅ Config loaded from {filepath}")
        except FileNotFoundError:
            logger.info(f"ℹ️ Config file not found: {filepath}, using defaults")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def get_timeframe(self) -> str:
        """Получить текущий таймфрейм индикатора"""
        return self.indicator_timeframe
    
    def set_timeframe(self, timeframe: str):
        """
        Установить таймфрейм индикатора
        Параметры:
            timeframe: '1' (1m), '5' (5m), '15' (15m), '60' (1h), '240' (4h), 'D' (1d)
        """
        valid_timeframes = ['1', '5', '15', '60', '240', 'D']
        if timeframe not in valid_timeframes:
            logger.warning(f"Invalid timeframe {timeframe}, using default 240 (4h)")
            timeframe = '240'
        
        self.indicator_timeframe = timeframe
        logger.info(f"✅ Timeframe set to {timeframe}")
    
    def get_timeframe_minutes(self) -> int:
        """Получить таймфрейм в минутах (для расчётов)"""
        timeframe_map = {
            '1': 1,
            '5': 5,
            '15': 15,
            '60': 60,
            '240': 240,
            'D': 1440,
        }
        return timeframe_map.get(self.indicator_timeframe, 240)
    
    def __repr__(self):
        return f"ScannerConfig(style={self.trading_style}, agg={self.aggressiveness}, auto={self.automation_mode})"


# Создать глобальный экземпляр конфигурации
scanner_config = ScannerConfig()
