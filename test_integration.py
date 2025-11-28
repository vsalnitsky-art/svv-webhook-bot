"""
Тест интеграции RSI+MFI индикатора и конфигурации сканера
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("ТЕСТ ИНТЕГРАЦИИ - ЭТАП 1")
print("="*70)

# Тест 1: Импорт модулей
print("\n1. Проверка импорта модулей...")
try:
    from rsi_mfi_indicator import RSIMFIIndicator
    print("   ✅ RSIMFIIndicator импортирован")
except Exception as e:
    print(f"   ❌ Ошибка импорта RSIMFIIndicator: {e}")
    sys.exit(1)

try:
    from scanner_config import ScannerConfig, scanner_config
    print("   ✅ ScannerConfig импортирован")
except Exception as e:
    print(f"   ❌ Ошибка импорта ScannerConfig: {e}")
    sys.exit(1)

try:
    from models import (Base, PositionAnalytics, PositionSnapshot, 
                       MarketCandidate, ScanHistory, AutoCloseDecision)
    print("   ✅ Новые модели импортированы")
except Exception as e:
    print(f"   ❌ Ошибка импорта моделей: {e}")
    sys.exit(1)

# Тест 2: Создание БД с новыми таблицами
print("\n2. Создание БД с новыми таблицами...")
try:
    from models import db_manager
    print("   ✅ База данных создана с новыми таблицами")
except Exception as e:
    print(f"   ❌ Ошибка создания БД: {e}")
    sys.exit(1)

# Тест 3: Проверка конфигурации
print("\n3. Проверка конфигурации...")
try:
    config = ScannerConfig()
    print(f"   ✅ Конфигурация создана: {config}")
    print(f"      - Стиль торговли: {config.trading_style}")
    print(f"      - Агрессивность: {config.aggressiveness}")
    print(f"      - Автоматизация: {config.automation_mode}")
except Exception as e:
    print(f"   ❌ Ошибка создания конфигурации: {e}")
    sys.exit(1)

# Тест 4: Проверка параметров индикатора
print("\n4. Проверка параметров индикатора...")
try:
    params = config.get_indicator_params()
    print(f"   ✅ Параметры индикатора получены:")
    print(f"      - RSI length: {params['rsi_length']}")
    print(f"      - Oversold: {params['oversold']}")
    print(f"      - Overbought: {params['overbought']}")
    print(f"      - MFI length: {params['mfi_length']}")
    print(f"      - Fast MFI EMA: {params['fast_mfi_ema']}")
    print(f"      - Slow MFI EMA: {params['slow_mfi_ema']}")
except Exception as e:
    print(f"   ❌ Ошибка получения параметров: {e}")
    sys.exit(1)

# Тест 5: Создание индикатора с параметрами
print("\n5. Создание индикатора с параметрами...")
try:
    indicator = RSIMFIIndicator(**params)
    print(f"   ✅ Индикатор создан успешно")
except Exception as e:
    print(f"   ❌ Ошибка создания индикатора: {e}")
    sys.exit(1)

# Тест 6: Обработка тестовых данных
print("\n6. Тест обработки данных...")
try:
    # Создание тестовых OHLCV данных
    dates = pd.date_range(datetime.now() - timedelta(hours=200), 
                         periods=200, freq='1h')
    np.random.seed(42)
    close_prices = 45000 + np.cumsum(np.random.randn(200) * 50)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(200) * 10,
        'high': close_prices + abs(np.random.randn(200) * 30),
        'low': close_prices - abs(np.random.randn(200) * 30),
        'close': close_prices,
        'volume': np.random.randint(100000, 1000000, 200)
    }, index=dates)
    
    print(f"   ✅ Тестовые данные созданы: {len(df)} свечей")
    
    # Обработка через индикатор
    result_df, alerts = indicator.process_data(df, symbol="BTCUSDT")
    print(f"   ✅ Данные обработаны: {len(result_df)} записей")
    print(f"   ✅ Алертов: {len(alerts)}")
    
    # Получение последних сигналов
    latest = indicator.get_latest_signals(result_df)
    print(f"   ✅ Последние сигналы:")
    print(f"      - RSI: {latest['rsi_value']:.2f}")
    print(f"      - MFI trend: {latest['mfi_trend']}")
    print(f"      - Last signal: {latest['last_signal']}")
    print(f"      - Strong bullish: {latest['strong_bullish_signal']}")
    print(f"      - Strong bearish: {latest['strong_bearish_signal']}")
    
except Exception as e:
    print(f"   ❌ Ошибка обработки данных: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Тест 7: Проверка пресетов
print("\n7. Проверка пресетов...")
try:
    # Скальпинг
    config.update_trading_style('scalping')
    params_scalp = config.get_indicator_params()
    print(f"   ✅ Скальпинг: RSI={params_scalp['rsi_length']}, levels={params_scalp['oversold']}/{params_scalp['overbought']}")
    
    # Свинг
    config.update_trading_style('swing')
    params_swing = config.get_indicator_params()
    print(f"   ✅ Свинг: RSI={params_swing['rsi_length']}, levels={params_swing['oversold']}/{params_swing['overbought']}")
    
    # Возврат к дейтрейдингу
    config.update_trading_style('daytrading')
    print(f"   ✅ Возврат к дейтрейдингу")
    
except Exception as e:
    print(f"   ❌ Ошибка проверки пресетов: {e}")
    sys.exit(1)

# Тест 8: Проверка Risk Management параметров
print("\n8. Проверка Risk Management...")
try:
    risk_params = config.get_risk_params()
    print(f"   ✅ Risk Management параметры:")
    print(f"      - Max positions: {risk_params['max_positions']}")
    print(f"      - Position size: {risk_params['max_position_size_percent']}%")
    print(f"      - Daily loss limit: {risk_params['daily_loss_limit_percent']}%")
    print(f"      - Max leverage: {risk_params['max_leverage']}x")
except Exception as e:
    print(f"   ❌ Ошибка Risk Management: {e}")
    sys.exit(1)

# Тест 9: Проверка Auto-Close параметров
print("\n9. Проверка Auto-Close параметров...")
try:
    auto_close_params = config.get_auto_close_params()
    print(f"   ✅ Auto-Close параметры:")
    print(f"      - Enabled: {auto_close_params['enabled']}")
    print(f"      - Use strong signals: {auto_close_params['use_strong_signals']}")
    print(f"      - Confirm with MFI: {auto_close_params['confirm_with_mfi']}")
    print(f"      - Min hold time: {auto_close_params['min_hold_time']}s")
except Exception as e:
    print(f"   ❌ Ошибка Auto-Close: {e}")
    sys.exit(1)

# Тест 10: Проверка Scanner параметров
print("\n10. Проверка Scanner параметров...")
try:
    scanner_params = config.get_scanner_params()
    print(f"   ✅ Scanner параметры:")
    print(f"      - Enabled: {scanner_params['enabled']}")
    print(f"      - Scan interval: {scanner_params['scan_interval']}s")
    print(f"      - Min volume: ${scanner_params['min_volume_24h']:,}")
    print(f"      - Top candidates: {scanner_params['top_candidates_count']}")
except Exception as e:
    print(f"   ❌ Ошибка Scanner: {e}")
    sys.exit(1)

# Итоги
print("\n" + "="*70)
print("ИТОГИ ТЕСТИРОВАНИЯ ЭТАПА 1")
print("="*70)
print("✅ Все тесты пройдены успешно!")
print()
print("Готово к реализации:")
print("  ✅ RSI+MFI индикатор интегрирован")
print("  ✅ Конфигурация с пресетами работает")
print("  ✅ База данных расширена новыми таблицами")
print("  ✅ Параметры настроены (дейтрейдинг, авто, полуавтомат)")
print()
print("Следующий этап: Position Monitor (автозакрытие позиций)")
print("="*70)
