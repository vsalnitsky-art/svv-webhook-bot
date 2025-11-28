"""
Advanced Order Strategy Configuration v3.0
==========================================
Розширена система стратегій для TP та SL з підтримкою TradingView
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OrderStrategyConfig:
    """Конфігурація стратегій відкриття ордеру"""
    
    # ==================== TP СТРАТЕГІЇ ====================
    
    TP_STRATEGIES = {
        'tradingview': {
            'name': 'Дані з TradingView',
            'name_ua': 'Дані з TradingView',
            'description': 'Використовувати TP з JSON сигналу TradingView',
            'description_ua': 'TP встановлюється на основі даних з TradingView (takeProfitPercent або tpLevels з JSON)',
            'enabled': True,
            'source': 'json',  # Джерело даних
            'targets': []  # Будуть з JSON
        },
        
        'none': {
            'name': 'Без TP',
            'name_ua': 'Без TP',
            'description': 'Не встановлювати Take Profit',
            'description_ua': 'TP не встановлюється, закриття тільки вручну або по сигналу',
            'enabled': False,
            'source': 'manual',
            'targets': []
        },
        
        'single': {
            'name': 'Single TP',
            'name_ua': 'Одиночний TP',
            'description': 'One take profit target',
            'description_ua': 'Один рівень Take Profit на всю позицію',
            'enabled': True,
            'source': 'preset',
            'targets': [
                {'percent': 1.5, 'quantity_percent': 100}
            ]
        },
        
        'conservative': {
            'name': 'Conservative',
            'name_ua': 'Консервативний',
            'description': 'Quick profit + remaining position',
            'description_ua': 'Швидкий прибуток + залишок позиції',
            'enabled': True,
            'source': 'preset',
            'targets': [
                {'percent': 0.8, 'quantity_percent': 50},
                {'percent': 2.0, 'quantity_percent': 50}
            ]
        },
        
        'balanced': {
            'name': 'Balanced',
            'name_ua': 'Збалансований',
            'description': 'Step-by-step profit taking',
            'description_ua': 'Поступове фіксування прибутку',
            'enabled': True,
            'source': 'preset',
            'targets': [
                {'percent': 1.0, 'quantity_percent': 33},
                {'percent': 2.0, 'quantity_percent': 33},
                {'percent': 3.5, 'quantity_percent': 34}
            ]
        },
        
        'scalper': {
            'name': 'Scalper',
            'name_ua': 'Скальпер',
            'description': 'Quick small profits',
            'description_ua': 'Швидкі невеликі прибутки',
            'enabled': True,
            'source': 'preset',
            'targets': [
                {'percent': 0.3, 'quantity_percent': 50},
                {'percent': 0.8, 'quantity_percent': 50}
            ]
        },
        
        'aggressive': {
            'name': 'Aggressive',
            'name_ua': 'Агресивний',
            'description': 'Maximum profit potential',
            'description_ua': 'Максимальний потенціал прибутку',
            'enabled': True,
            'source': 'preset',
            'targets': [
                {'percent': 0.5, 'quantity_percent': 25},
                {'percent': 1.5, 'quantity_percent': 25},
                {'percent': 3.0, 'quantity_percent': 25},
                {'percent': 5.0, 'quantity_percent': 25}
            ]
        }
    }
    
    # ==================== SL СТРАТЕГІЇ ====================
    
    SL_STRATEGIES = {
        'tradingview': {
            'name': 'Дані з TradingView',
            'name_ua': 'Дані з TradingView',
            'description': 'Use SL from TradingView signal',
            'description_ua': 'SL встановлюється на основі даних з TradingView (stopLossPercent з JSON)',
            'enabled': True,
            'source': 'json',
            'mode': 'fixed',
            'percent': None  # Буде з JSON
        },
        
        'none': {
            'name': 'Без SL',
            'name_ua': 'Без SL',
            'description': 'No stop loss',
            'description_ua': 'SL не встановлюється (небезпечно!)',
            'enabled': False,
            'source': 'manual',
            'mode': 'none',
            'percent': None
        },
        
        'tight': {
            'name': 'Tight SL',
            'name_ua': 'Жорсткий SL',
            'description': 'Small stop loss',
            'description_ua': 'Малий стоп-лосс (1.0%)',
            'enabled': True,
            'source': 'preset',
            'mode': 'fixed',
            'percent': 1.0
        },
        
        'normal': {
            'name': 'Normal SL',
            'name_ua': 'Нормальний SL',
            'description': 'Standard stop loss',
            'description_ua': 'Стандартний стоп-лосс (2.0%)',
            'enabled': True,
            'source': 'preset',
            'mode': 'fixed',
            'percent': 2.0
        },
        
        'wide': {
            'name': 'Wide SL',
            'name_ua': 'Широкий SL',
            'description': 'Larger stop loss',
            'description_ua': 'Більший стоп-лосс (3.5%)',
            'enabled': True,
            'source': 'preset',
            'mode': 'fixed',
            'percent': 3.5
        },
        
        'breakeven_on_tp1': {
            'name': 'Breakeven on TP1',
            'name_ua': 'Беззбиток після TP1',
            'description': 'Move to breakeven when TP1 hits',
            'description_ua': 'Переміщення SL в беззбиток при досягненні першого TP',
            'enabled': True,
            'source': 'preset',
            'mode': 'dynamic',
            'initial_percent': 2.0,
            'breakeven_trigger': 'tp1',  # Після першого TP
            'breakeven_offset': 0.1  # 0.1% вище входу
        },
        
        'trailing': {
            'name': 'Trailing SL',
            'name_ua': 'Трейлінг SL',
            'description': 'Trailing stop loss',
            'description_ua': 'SL слідує за ціною',
            'enabled': True,
            'source': 'preset',
            'mode': 'trailing',
            'initial_percent': 2.0,
            'trailing_percent': 1.0,  # Відстань від поточної ціни
            'activation_percent': 1.5  # Активація після +1.5% прибутку
        }
    }
    
    def __init__(self):
        """Ініціалізація"""
        # Дефолтні стратегії
        self.tp_strategy = 'tradingview'  # ⭐ За замовчуванням TradingView
        self.sl_strategy = 'tradingview'  # ⭐ За замовчуванням TradingView
        
        # Комбінована стратегія (приклад)
        self.combined_strategy = None  # Можна зберегти комбінацію
        
        logger.info("✅ Order Strategy Config initialized")
        logger.info(f"   Default TP: {self.tp_strategy}")
        logger.info(f"   Default SL: {self.sl_strategy}")
    
    def set_tp_strategy(self, strategy_name: str):
        """Встановити TP стратегію"""
        if strategy_name in self.TP_STRATEGIES:
            self.tp_strategy = strategy_name
            logger.info(f"✅ TP Strategy set to: {strategy_name}")
        else:
            logger.error(f"❌ Unknown TP strategy: {strategy_name}")
    
    def set_sl_strategy(self, strategy_name: str):
        """Встановити SL стратегію"""
        if strategy_name in self.SL_STRATEGIES:
            self.sl_strategy = strategy_name
            logger.info(f"✅ SL Strategy set to: {strategy_name}")
        else:
            logger.error(f"❌ Unknown SL strategy: {strategy_name}")
    
    def get_tp_strategy(self) -> Dict[str, Any]:
        """Отримати поточну TP стратегію"""
        return self.TP_STRATEGIES.get(self.tp_strategy, self.TP_STRATEGIES['tradingview'])
    
    def get_sl_strategy(self) -> Dict[str, Any]:
        """Отримати поточну SL стратегію"""
        return self.SL_STRATEGIES.get(self.sl_strategy, self.SL_STRATEGIES['tradingview'])
    
    def set_combined_strategy(self, tp_strategy: str, sl_strategy: str, name: Optional[str] = None):
        """
        Встановити комбіновану стратегію
        
        Приклад:
            tp_strategy='scalper' - Скальпер (2 рівні TP)
            sl_strategy='breakeven_on_tp1' - SL в беззбиток після TP1
        """
        if tp_strategy not in self.TP_STRATEGIES:
            logger.error(f"❌ Invalid TP strategy: {tp_strategy}")
            return False
        
        if sl_strategy not in self.SL_STRATEGIES:
            logger.error(f"❌ Invalid SL strategy: {sl_strategy}")
            return False
        
        self.tp_strategy = tp_strategy
        self.sl_strategy = sl_strategy
        
        self.combined_strategy = {
            'name': name or f"{tp_strategy} + {sl_strategy}",
            'tp': tp_strategy,
            'sl': sl_strategy
        }
        
        logger.info(f"✅ Combined strategy set:")
        logger.info(f"   TP: {tp_strategy}")
        logger.info(f"   SL: {sl_strategy}")
        
        return True
    
    def get_tp_config_for_order(self, json_data: Dict = None) -> Dict[str, Any]:
        """
        Отримати конфігурацію TP для ордера
        
        Args:
            json_data: Дані з TradingView (якщо є)
        """
        tp_strategy = self.get_tp_strategy()
        
        if tp_strategy['source'] == 'json' and json_data:
            # Використати дані з TradingView
            if 'tpLevels' in json_data:
                return {
                    'source': 'tradingview',
                    'mode': 'custom_levels',
                    'levels': json_data['tpLevels']
                }
            elif 'takeProfitPercent' in json_data:
                return {
                    'source': 'tradingview',
                    'mode': 'single',
                    'levels': [{'percent': float(json_data['takeProfitPercent']), 'quantity_percent': 100}]
                }
            else:
                # Немає TP в JSON, використати preset
                logger.warning("⚠️ TradingView selected but no TP in JSON, using fallback")
                return {
                    'source': 'fallback',
                    'mode': 'preset',
                    'levels': self.TP_STRATEGIES['balanced']['targets']
                }
        else:
            # Використати preset стратегію
            return {
                'source': 'preset',
                'mode': tp_strategy['name_ua'],
                'levels': tp_strategy['targets']
            }
    
    def get_sl_config_for_order(self, json_data: Dict = None) -> Dict[str, Any]:
        """
        Отримати конфігурацію SL для ордера
        
        Args:
            json_data: Дані з TradingView (якщо є)
        """
        sl_strategy = self.get_sl_strategy()
        
        if sl_strategy['source'] == 'json' and json_data:
            # Використати дані з TradingView
            if 'stopLossPercent' in json_data:
                return {
                    'source': 'tradingview',
                    'mode': 'fixed',
                    'percent': float(json_data['stopLossPercent']),
                    'dynamic': False
                }
            else:
                # Немає SL в JSON, використати preset
                logger.warning("⚠️ TradingView selected but no SL in JSON, using fallback")
                return {
                    'source': 'fallback',
                    'mode': 'fixed',
                    'percent': 2.0,
                    'dynamic': False
                }
        else:
            # Використати preset стратегію
            config = {
                'source': 'preset',
                'mode': sl_strategy['mode'],
                'percent': sl_strategy.get('percent'),
                'dynamic': sl_strategy['mode'] in ['dynamic', 'trailing']
            }
            
            # Додаткові параметри для динамічних стратегій
            if sl_strategy['mode'] == 'dynamic':
                config['initial_percent'] = sl_strategy.get('initial_percent', 2.0)
                config['breakeven_trigger'] = sl_strategy.get('breakeven_trigger', 'tp1')
                config['breakeven_offset'] = sl_strategy.get('breakeven_offset', 0.1)
            elif sl_strategy['mode'] == 'trailing':
                config['initial_percent'] = sl_strategy.get('initial_percent', 2.0)
                config['trailing_percent'] = sl_strategy.get('trailing_percent', 1.0)
                config['activation_percent'] = sl_strategy.get('activation_percent', 1.5)
            
            return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Експорт в dict"""
        return {
            'tp_strategy': self.tp_strategy,
            'sl_strategy': self.sl_strategy,
            'combined_strategy': self.combined_strategy
        }


# Глобальний екземпляр
order_strategy_config = OrderStrategyConfig()
