#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests для критичних компонентів бота
Запуск: pytest tests.py -v --tb=short
"""
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from utils import (
    validate_webhook_data, validate_stop_loss, 
    safe_float, safe_int, metrics
)

# ===== ВАЛІДАЦІЯ ВХІДНИХ ДАНИХ =====

class TestWebhookValidation:
    """Тести для валідації webhook даних"""
    
    def test_valid_buy_signal(self):
        """Валідний сигнал Buy"""
        data = {
            'action': 'Buy',
            'symbol': 'BTCUSDT',
            'riskPercent': 2.0,
            'leverage': 20,
            'sl_price': 30000.0
        }
        result = validate_webhook_data(data)
        assert result['action'] == 'Buy'
        assert result['symbol'] == 'BTCUSDT'
        assert result['riskPercent'] == 2.0
    
    def test_valid_close_signal(self):
        """Валідний сигнал Close"""
        data = {
            'action': 'Close',
            'symbol': 'BTCUSDT',
            'direction': 'Long'
        }
        result = validate_webhook_data(data)
        assert result['action'] == 'Close'
        assert result['direction'] == 'Long'
    
    def test_invalid_action(self):
        """Невалідна дія"""
        data = {
            'action': 'Invalid',
            'symbol': 'BTCUSDT',
            'riskPercent': 2.0,
            'leverage': 20
        }
        with pytest.raises(ValueError) as exc:
            validate_webhook_data(data)
        assert 'Invalid action' in str(exc.value)
    
    def test_missing_risk_percent(self):
        """Відсутній riskPercent для Buy"""
        data = {
            'action': 'Buy',
            'symbol': 'BTCUSDT',
            'leverage': 20
        }
        with pytest.raises(ValueError) as exc:
            validate_webhook_data(data)
        assert 'riskPercent required' in str(exc.value)
    
    def test_invalid_symbol(self):
        """Невалідний символ (не закінчується на USDT)"""
        data = {
            'action': 'Buy',
            'symbol': 'BTC',
            'riskPercent': 2.0,
            'leverage': 20
        }
        with pytest.raises(ValueError) as exc:
            validate_webhook_data(data)
        assert 'Symbol must end with' in str(exc.value)
    
    def test_risk_out_of_range(self):
        """Risk поза допустимим діапазоном"""
        data = {
            'action': 'Buy',
            'symbol': 'BTCUSDT',
            'riskPercent': 15.0,  # > 10%
            'leverage': 20
        }
        with pytest.raises(ValueError) as exc:
            validate_webhook_data(data)
        assert 'must be 0.1-10%' in str(exc.value)
    
    def test_leverage_out_of_range(self):
        """Leverage поза допустимим діапазоном"""
        data = {
            'action': 'Buy',
            'symbol': 'BTCUSDT',
            'riskPercent': 2.0,
            'leverage': 150  # > 100x
        }
        with pytest.raises(ValueError) as exc:
            validate_webhook_data(data)
        assert 'must be 1-100x' in str(exc.value)

# ===== ВАЛІДАЦІЯ STOP LOSS =====

class TestStopLossValidation:
    """Тести для валідації Stop Loss"""
    
    def test_valid_long_sl(self):
        """Валідний SL для лонга (нижче ціни входу)"""
        assert validate_stop_loss(30000, 35000, 'Buy') is True
    
    def test_invalid_long_sl_above_entry(self):
        """Невалідний SL для лонга (вище ціни входу)"""
        assert validate_stop_loss(36000, 35000, 'Buy') is False
    
    def test_valid_short_sl(self):
        """Валідний SL для шорта (вище ціни входу)"""
        assert validate_stop_loss(40000, 35000, 'Sell') is True
    
    def test_invalid_short_sl_below_entry(self):
        """Невалідний SL для шорта (нижче ціни входу)"""
        assert validate_stop_loss(30000, 35000, 'Sell') is False
    
    def test_sl_zero_or_negative(self):
        """SL <= 0 завжди невалідний"""
        assert validate_stop_loss(0, 35000, 'Buy') is False
        assert validate_stop_loss(-100, 35000, 'Buy') is False

# ===== УТИЛІТИ =====

class TestUtilities:
    """Тести для допоміжних функцій"""
    
    def test_safe_float_valid(self):
        """Конвертація валідного числа у float"""
        assert safe_float('123.45') == 123.45
        assert safe_float(123.45) == 123.45
        assert safe_float(123) == 123.0
    
    def test_safe_float_invalid(self):
        """Конвертація невалідного значення у float"""
        assert safe_float('invalid') == 0.0
        assert safe_float('invalid', 99.0) == 99.0
        assert safe_float(None) == 0.0
    
    def test_safe_int_valid(self):
        """Конвертація валідного числа у int"""
        assert safe_int('123') == 123
        assert safe_int(123.45) == 123
        assert safe_int(123) == 123
    
    def test_safe_int_invalid(self):
        """Конвертація невалідного значення у int"""
        assert safe_int('invalid') == 0
        assert safe_int('invalid', 99) == 99
        assert safe_int(None) == 0

# ===== МЕТРИКИ =====

class TestMetrics:
    """Тести для метрик"""
    
    def test_metrics_initialization(self):
        """Ініціалізація метрик"""
        test_metrics = metrics.__class__()
        stats = test_metrics.get_stats()
        assert stats['trades_opened'] == 0
        assert stats['trades_closed'] == 0
        assert stats['total_pnl'] == 0.0
    
    def test_log_trade_opened(self):
        """Логування відкритої позиції"""
        test_metrics = metrics.__class__()
        test_metrics.log_trade_opened('BTCUSDT', 1.0, 35000)
        stats = test_metrics.get_stats()
        assert stats['trades_opened'] == 1
    
    def test_log_trade_closed(self):
        """Логування закритої позиції"""
        test_metrics = metrics.__class__()
        test_metrics.log_trade_closed('BTCUSDT', 100.0)
        stats = test_metrics.get_stats()
        assert stats['trades_closed'] == 1
        assert stats['total_pnl'] == 100.0
    
    def test_success_rate(self):
        """Розрахунок success rate"""
        test_metrics = metrics.__class__()
        test_metrics.log_trade_opened('BTCUSDT', 1.0, 35000)
        test_metrics.log_trade_opened('ETHUSDT', 10.0, 2000)
        test_metrics.log_trade_closed('BTCUSDT', 100.0)
        stats = test_metrics.get_stats()
        # 1 closed / 2 opened = 50%
        assert stats['success_rate'] == 50.0

# ===== КОНФІГ =====

class TestEnvironmentVariables:
    """Тести для Environment Variables"""
    
    def test_env_variables_exist(self):
        """Базова перевірка ENV змінних"""
        # Додаємо лише ті змінні, які можуть бути в тесті
        port = os.environ.get('PORT', 10000)
        assert int(port) > 0
    
    def test_render_detection(self):
        """Перевірка окружени"""
        render = os.environ.get('RENDER', 'false')
        assert render in ['true', 'false', None]


# ===== ІНТЕГРАЦІЙНІ ТЕСТИ =====

class TestBotIntegration:
    """Інтеграційні тести для основних операцій"""
    
    @patch('bot.bot_instance')
    def test_place_order_with_valid_data(self, mock_bot):
        """Розміщення ордера з валідними даними"""
        from bot import BybitTradingBot
        
        bot = BybitTradingBot()
        bot.session = MagicMock()
        
        # Мокуємо API відповіді
        bot.session.get_wallet_balance.return_value = {
            'retCode': 0,
            'result': {
                'list': [{
                    'coin': [{'coin': 'USDT', 'walletBalance': '100'}]
                }]
            }
        }
        
        # Перевіряємо, що валідація працює
        data = {
            'action': 'Buy',
            'symbol': 'BTCUSDT',
            'riskPercent': 2.0,
            'leverage': 20,
            'sl_price': 30000.0
        }
        
        # Це не буде помилка валідації
        try:
            validated = validate_webhook_data(data)
            assert validated is not None
        except ValueError:
            pytest.fail("Validation should not fail for valid data")

# ===== ЗАПУСК ТЕСТІВ =====

if __name__ == '__main__':
    import os
    pytest.main([__file__, '-v', '--tb=short'])
