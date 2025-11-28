"""
Take Profit Strategy Configuration v2.4
========================================
Гнучка система управління TP з різними стратегіями
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TPStrategyConfig:
    """Конфігурація стратегій Take Profit"""
    
    # ==================== СТРАТЕГІЇ TP ====================
    
    TP_STRATEGIES = {
        'none': {
            'name': 'Без TP',
            'name_ua': 'Без TP',
            'description': 'Не встановлювати Take Profit взагалі',
            'description_ua': 'TP не встановлюється, закриття тільки вручну або по сигналу',
            'enabled': False,
            'targets': []
        },
        
        'single': {
            'name': 'Single TP',
            'name_ua': 'Одиночний TP',
            'description': 'One take profit target',
            'description_ua': 'Один рівень Take Profit на всю позицію',
            'enabled': True,
            'targets': [
                {'percent': 1.5, 'quantity_percent': 100}  # 1.5% прибутку, закрити 100%
            ]
        },
        
        'conservative': {
            'name': 'Conservative (2 levels)',
            'name_ua': 'Консервативний (2 рівні)',
            'description': 'Quick small profit + remaining position',
            'description_ua': 'Швидкий невеликий прибуток + залишок позиції',
            'enabled': True,
            'targets': [
                {'percent': 0.8, 'quantity_percent': 50},   # 0.8% - закрити 50%
                {'percent': 2.0, 'quantity_percent': 50}    # 2.0% - закрити решту
            ]
        },
        
        'balanced': {
            'name': 'Balanced (3 levels)',
            'name_ua': 'Збалансований (3 рівні)',
            'description': 'Step-by-step profit taking',
            'description_ua': 'Поступове фіксування прибутку',
            'enabled': True,
            'targets': [
                {'percent': 1.0, 'quantity_percent': 33},   # 1.0% - закрити 33%
                {'percent': 2.0, 'quantity_percent': 33},   # 2.0% - закрити 33%
                {'percent': 3.5, 'quantity_percent': 34}    # 3.5% - закрити решту
            ]
        },
        
        'aggressive': {
            'name': 'Aggressive (4 levels)',
            'name_ua': 'Агресивний (4 рівні)',
            'description': 'Maximum profit potential',
            'description_ua': 'Максимальний потенціал прибутку',
            'enabled': True,
            'targets': [
                {'percent': 0.5, 'quantity_percent': 25},   # 0.5% - закрити 25%
                {'percent': 1.5, 'quantity_percent': 25},   # 1.5% - закрити 25%
                {'percent': 3.0, 'quantity_percent': 25},   # 3.0% - закрити 25%
                {'percent': 5.0, 'quantity_percent': 25}    # 5.0% - закрити решту
            ]
        },
        
        'scalper': {
            'name': 'Scalper (fast)',
            'name_ua': 'Скальпер (швидко)',
            'description': 'Quick small profits',
            'description_ua': 'Швидкі невеликі прибутки',
            'enabled': True,
            'targets': [
                {'percent': 0.3, 'quantity_percent': 50},   # 0.3% - закрити 50%
                {'percent': 0.8, 'quantity_percent': 50}    # 0.8% - закрити решту
            ]
        },
        
        'hodler': {
            'name': 'Hodler (patient)',
            'name_ua': 'Тримач (терплячий)',
            'description': 'Let winners run',
            'description_ua': 'Дати прибутковим угодам рости',
            'enabled': True,
            'targets': [
                {'percent': 3.0, 'quantity_percent': 30},   # 3.0% - закрити 30%
                {'percent': 6.0, 'quantity_percent': 30},   # 6.0% - закрити 30%
                {'percent': 10.0, 'quantity_percent': 40}   # 10.0% - закрити решту
            ]
        },
        
        'fibonacci': {
            'name': 'Fibonacci levels',
            'name_ua': 'Рівні Фібоначчі',
            'description': 'Based on Fib retracement',
            'description_ua': 'На основі рівнів Фібоначчі',
            'enabled': True,
            'targets': [
                {'percent': 0.618, 'quantity_percent': 25},  # 0.618% (Fib)
                {'percent': 1.618, 'quantity_percent': 25},  # 1.618% (Golden ratio)
                {'percent': 2.618, 'quantity_percent': 25},  # 2.618%
                {'percent': 4.236, 'quantity_percent': 25}   # 4.236%
            ]
        },
        
        'custom': {
            'name': 'Custom',
            'name_ua': 'Власний',
            'description': 'User defined levels',
            'description_ua': 'Налаштовані користувачем рівні',
            'enabled': True,
            'targets': [
                # Користувач може налаштувати свої рівні
                {'percent': 1.0, 'quantity_percent': 50},
                {'percent': 2.0, 'quantity_percent': 50}
            ]
        }
    }
    
    def __init__(self):
        """Ініціалізація з дефолтною стратегією"""
        self.current_strategy = 'balanced'  # За замовчуванням
        self.use_trailing_tp = False  # Трейлінг TP
        self.trailing_callback_percent = 0.5  # 0.5% callback для трейлінгу
        
        # Індивідуальні налаштування (перевизначають стратегію)
        self.custom_targets = []
        
        logger.info(f"✅ TP Strategy initialized: {self.current_strategy}")
    
    def get_strategy(self, strategy_name: str = None) -> Dict[str, Any]:
        """Отримати стратегію за назвою"""
        if strategy_name is None:
            strategy_name = self.current_strategy
        
        if strategy_name in self.TP_STRATEGIES:
            return self.TP_STRATEGIES[strategy_name]
        else:
            logger.warning(f"⚠️ Unknown strategy: {strategy_name}, using balanced")
            return self.TP_STRATEGIES['balanced']
    
    def get_targets(self, strategy_name: str = None) -> List[Dict[str, float]]:
        """Отримати цілі TP для стратегії"""
        strategy = self.get_strategy(strategy_name)
        
        # Якщо є custom targets, використовувати їх
        if strategy_name == 'custom' and self.custom_targets:
            return self.custom_targets
        
        return strategy['targets']
    
    def set_strategy(self, strategy_name: str):
        """Встановити стратегію"""
        if strategy_name in self.TP_STRATEGIES:
            self.current_strategy = strategy_name
            logger.info(f"✅ TP Strategy changed to: {strategy_name}")
        else:
            logger.error(f"❌ Invalid strategy: {strategy_name}")
    
    def set_custom_targets(self, targets: List[Dict[str, float]]):
        """Встановити власні рівні TP"""
        # Перевірка що сума quantity_percent = 100%
        total = sum(t['quantity_percent'] for t in targets)
        if abs(total - 100) > 0.01:
            logger.warning(f"⚠️ Custom targets total != 100%: {total}%")
        
        self.custom_targets = targets
        logger.info(f"✅ Custom TP targets set: {len(targets)} levels")
    
    def calculate_tp_prices(self, entry_price: float, side: str, 
                           strategy_name: str = None) -> List[Dict[str, Any]]:
        """
        Розрахувати ціни TP для позиції
        
        Args:
            entry_price: Ціна входу
            side: 'Buy' або 'Sell'
            strategy_name: Назва стратегії (None = поточна)
        
        Returns:
            List[{'price': float, 'quantity_percent': float, 'profit_percent': float}]
        """
        targets = self.get_targets(strategy_name)
        
        if not targets:
            return []
        
        tp_levels = []
        for target in targets:
            profit_pct = target['percent']
            qty_pct = target['quantity_percent']
            
            # Розрахунок ціни TP
            if side == 'Buy':
                # Long: TP вище entry
                tp_price = entry_price * (1 + profit_pct / 100)
            else:
                # Short: TP нижче entry
                tp_price = entry_price * (1 - profit_pct / 100)
            
            tp_levels.append({
                'price': tp_price,
                'quantity_percent': qty_pct,
                'profit_percent': profit_pct
            })
        
        return tp_levels
    
    def get_all_strategies_info(self) -> Dict[str, Dict]:
        """Отримати інформацію про всі стратегії"""
        return self.TP_STRATEGIES
    
    def to_dict(self) -> Dict[str, Any]:
        """Експорт конфігурації в dict"""
        return {
            'current_strategy': self.current_strategy,
            'use_trailing_tp': self.use_trailing_tp,
            'trailing_callback_percent': self.trailing_callback_percent,
            'custom_targets': self.custom_targets,
            'available_strategies': list(self.TP_STRATEGIES.keys())
        }


# Глобальний екземпляр
tp_config = TPStrategyConfig()
