#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 SMART EXIT - ГОТОВІ КОНФІГУРАЦІЇ

3 готові профілі для різних стилів торгівлі
"""

from smart_exit_strategy import smart_exit

# ====================================================================
# ПРОФІЛЬ 1: КОНСЕРВАТИВНИЙ (безпека)
# ====================================================================

def configure_conservative():
    """
    🛡️ КОНСЕРВАТИВНИЙ ПРОФІЛЬ
    
    - Мало ризику, но менше прибутку
    - Trailing Stop: -0.3% (жорстко)
    - Дивергенція: 3 свічки (менш чутливе)
    - RSI Threshold: 75 (глибока перекупленість)
    
    Ідеально для:
    - Нові трейдери
    - Малий баланс
    - Дніжні фінансові цілі
    """
    
    smart_exit.TRAILING_STOP_PERCENT = -0.003      # -0.3%
    smart_exit.MIN_DIVERGENCE_CANDLES = 3          # 3 свічки
    smart_exit.RSI_THRESHOLD = 75                  # Глибока перекупленість
    
    return {
        'name': 'CONSERVATIVE',
        'trailing_stop_percent': -0.003,
        'min_divergence_candles': 3,
        'rsi_threshold': 75,
        'expected_profit': '+0.5% - +1.5%',
        'expected_loss': '-0.3%',
        'style': 'Safety First'
    }


# ====================================================================
# ПРОФІЛЬ 2: ЗБАЛАНСОВАНИЙ (рекомендується)
# ====================================================================

def configure_balanced():
    """
    ⚖️ ЗБАЛАНСОВАНИЙ ПРОФІЛЬ (за замовчуванням)
    
    - Баланс между ризиком и прибутком
    - Trailing Stop: -0.5% (нормально)
    - Дивергенція: 2 свічки (стабільне)
    - RSI Threshold: 70 (стандарт)
    
    Ідеально для:
    - Більшість трейдерів
    - Середній баланс
    - Регулярні доходи
    
    🏆 РЕКОМЕНДУЄТЬСЯ!
    """
    
    smart_exit.TRAILING_STOP_PERCENT = -0.005      # -0.5%
    smart_exit.MIN_DIVERGENCE_CANDLES = 2          # 2 свічки
    smart_exit.RSI_THRESHOLD = 70                  # Стандарт
    
    return {
        'name': 'BALANCED',
        'trailing_stop_percent': -0.005,
        'min_divergence_candles': 2,
        'rsi_threshold': 70,
        'expected_profit': '+1.0% - +2.5%',
        'expected_loss': '-0.5%',
        'style': 'Balanced Risk/Reward'
    }


# ====================================================================
# ПРОФІЛЬ 3: АГРЕСИВНИЙ (максимум прибутку)
# ====================================================================

def configure_aggressive():
    """
    🚀 АГРЕСИВНИЙ ПРОФІЛЬ
    
    - Максимум прибутку, але вишчий ризик
    - Trailing Stop: -1.0% (м'яко)
    - Дивергенція: 1 свічка (дуже чутливе)
    - RSI Threshold: 65 (перекупленість)
    
    Ідеально для:
    - Досвідчені трейдери
    - Великий баланс
    - Висока толерантність до ризику
    
    ⚠️ РИЗИКУЄ!
    """
    
    smart_exit.TRAILING_STOP_PERCENT = -0.010      # -1.0%
    smart_exit.MIN_DIVERGENCE_CANDLES = 1          # 1 свічка
    smart_exit.RSI_THRESHOLD = 65                  # Перекупленість
    
    return {
        'name': 'AGGRESSIVE',
        'trailing_stop_percent': -0.010,
        'min_divergence_candles': 1,
        'rsi_threshold': 65,
        'expected_profit': '+2.0% - +5.0%',
        'expected_loss': '-1.0%',
        'style': 'Maximize Profit'
    }


# ====================================================================
# ПРОФІЛЬ 4: СКАЛЬПІНГ (швидкі угоди)
# ====================================================================

def configure_scalping():
    """
    ⚡ СКАЛЬПІНГ ПРОФІЛЬ
    
    - Короткі угоди з малим прибутком
    - Trailing Stop: -0.2% (дуже жорстко)
    - Дивергенція: 1 свічка (максимум чутливо)
    - RSI Threshold: 72 (вищий поріг)
    
    Ідеально для:
    - Скальпери
    - Бот-трейдинг
    - Короткотривалі позиції
    
    ⚡ ШВИДКО!
    """
    
    smart_exit.TRAILING_STOP_PERCENT = -0.002      # -0.2%
    smart_exit.MIN_DIVERGENCE_CANDLES = 1          # 1 свічка
    smart_exit.RSI_THRESHOLD = 72                  # Вищий поріг
    
    return {
        'name': 'SCALPING',
        'trailing_stop_percent': -0.002,
        'min_divergence_candles': 1,
        'rsi_threshold': 72,
        'expected_profit': '+0.2% - +0.8%',
        'expected_loss': '-0.2%',
        'style': 'Quick Profits'
    }


# ====================================================================
# ПРОФІЛЬ 5: SWING TRADING (середньострокові)
# ====================================================================

def configure_swing():
    """
    📊 SWING TRADING ПРОФІЛЬ
    
    - Середньострокові позиції (часи)
    - Trailing Stop: -1.5% (м'яко)
    - Дивергенція: 3 свічки (дуже стабільне)
    - RSI Threshold: 68 (м'яке)
    
    Ідеально для:
    - Swing трейдери
    - Позиції на годинах/днях
    - Більші ходи ціни
    
    📈 ДОВГОСТРОК
    """
    
    smart_exit.TRAILING_STOP_PERCENT = -0.015      # -1.5%
    smart_exit.MIN_DIVERGENCE_CANDLES = 3          # 3 свічки
    smart_exit.RSI_THRESHOLD = 68                  # М'яке
    
    return {
        'name': 'SWING',
        'trailing_stop_percent': -0.015,
        'min_divergence_candles': 3,
        'rsi_threshold': 68,
        'expected_profit': '+2.0% - +5.0%',
        'expected_loss': '-1.5%',
        'style': 'Medium Term'
    }


# ====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ====================================================================

def print_config(config):
    """Вивести конфіг красиво"""
    
    print("\n" + "=" * 70)
    print(f"📋 SMART EXIT PROFILE: {config['name']}")
    print("=" * 70)
    print(f"Style:                    {config['style']}")
    print(f"Trailing Stop:            {config['trailing_stop_percent'] * 100}%")
    print(f"Min Divergence Candles:   {config['min_divergence_candles']}")
    print(f"RSI Threshold:            {config['rsi_threshold']}")
    print(f"Expected Profit Range:    {config['expected_profit']}")
    print(f"Expected Max Loss:        {config['expected_loss']}")
    print("=" * 70 + "\n")


def print_all_profiles():
    """Вивести всі профілі"""
    
    profiles = [
        configure_conservative(),
        configure_balanced(),
        configure_aggressive(),
        configure_scalping(),
        configure_swing()
    ]
    
    print("\n" + "🎯" * 35)
    print("SMART EXIT - ВСІ ДОСТУПНІ ПРОФІЛІ")
    print("🎯" * 35 + "\n")
    
    for i, config in enumerate(profiles, 1):
        print(f"{i}. {config['name']:12} | TS: {config['trailing_stop_percent']*100:5.1f}% | "
              f"DIV: {config['min_divergence_candles']} | RSI: {config['rsi_threshold']:2d} | "
              f"Profit: {config['expected_profit']:15} | {config['style']}")
    
    print()


def get_profile_by_name(name):
    """Отримати профіл за назвою"""
    
    profiles = {
        'conservative': configure_conservative,
        'balanced': configure_balanced,
        'aggressive': configure_aggressive,
        'scalping': configure_scalping,
        'swing': configure_swing
    }
    
    profile_func = profiles.get(name.lower())
    if profile_func:
        return profile_func()
    else:
        print(f"❌ Profile '{name}' not found!")
        print(f"Available: {', '.join(profiles.keys())}")
        return None


# ====================================================================
# ПРИКЛАД ВИКОРИСТАННЯ
# ====================================================================

if __name__ == "__main__":
    
    # Показати всі профілі
    print_all_profiles()
    
    # Вибрати профіль
    print("=" * 70)
    print("📌 ВИБІР ПРОФІЛЮ")
    print("=" * 70)
    
    choice = input("Виберіть профіль (conservative/balanced/aggressive/scalping/swing): ").strip().lower()
    
    if choice == "conservative":
        config = configure_conservative()
    elif choice == "balanced":
        config = configure_balanced()
    elif choice == "aggressive":
        config = configure_aggressive()
    elif choice == "scalping":
        config = configure_scalping()
    elif choice == "swing":
        config = configure_swing()
    else:
        print("❌ Невірний вибір! Використовую BALANCED за замовчуванням")
        config = configure_balanced()
    
    print_config(config)
    
    # Показати поточні налаштування
    print("✅ ПОТОЧНІ НАЛАШТУВАННЯ SMART EXIT:")
    print(f"   Trailing Stop: {smart_exit.TRAILING_STOP_PERCENT * 100}%")
    print(f"   Min Divergence: {smart_exit.MIN_DIVERGENCE_CANDLES} свічок")
    print(f"   RSI Threshold: {smart_exit.RSI_THRESHOLD}")
    
    # Готові до запуску!
    print("\n🚀 Готово до запуску! Профіль '{0}' активний.".format(config['name']))
