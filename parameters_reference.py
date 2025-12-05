#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 ПОВНА ТАБЛИЦЯ ПАРАМЕТРІВ ДЛЯ ВСІХ ТАЙМФРЕЙМІВ І ПРОФІЛІВ

Це допоміжний файл для вивілення всієї інформації
"""

from timeframe_parameters import TimeframeParameters

def print_parameters_table():
    """Вивести таблицю всіх параметрів"""
    
    timeframes = TimeframeParameters.get_all_timeframes()
    profiles = TimeframeParameters.get_all_profiles()
    
    print("\n" + "=" * 150)
    print("📊 ТАБЛИЦЯ ПОКАЗНИКІВ: ВСІ ТАЙМФРЕЙМИ × ВСІ ПРОФІЛІ")
    print("=" * 150)
    
    # ТАБЛИЦЯ 1: RSI THRESHOLD
    print("\n1️⃣ RSI THRESHOLD (для LONG позицій)")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            print(f"| {config['rsi_threshold_long']:<15}", end="")
        print("|")
    
    # ТАБЛИЦЯ 2: TRAILING STOP
    print("\n\n2️⃣ TRAILING STOP PERCENT")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            ts = config['trailing_stop_percent'] * 100
            print(f"| {ts:>6.2f}% {'':<6}", end="")
        print("|")
    
    # ТАБЛИЦЯ 3: MIN DIVERGENCE
    print("\n\n3️⃣ MIN DIVERGENCE CANDLES")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            print(f"| {config['min_divergence_candles']:<15}", end="")
        print("|")
    
    # ТАБЛИЦЯ 4: STOP LOSS
    print("\n\n4️⃣ STOP LOSS PERCENT")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            sl = config['stop_loss_percent']
            print(f"| {sl:>6.2f}% {'':<6}", end="")
        print("|")
    
    # ТАБЛИЦЯ 5: TAKE PROFIT
    print("\n\n5️⃣ TAKE PROFIT PERCENT")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            tp = config['take_profit_percent']
            print(f"| {tp:>6.2f}% {'':<6}", end="")
        print("|")
    
    # ТАБЛИЦЯ 6: RSI PERIOD
    print("\n\n6️⃣ RSI PERIOD")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            print(f"| {config['rsi_period']:<15}", end="")
        print("|")
    
    # ТАБЛИЦЯ 7: ОЧІКУВАНИЙ ПРИБУТОК
    print("\n\n7️⃣ ОЧІКУВАНИЙ ПРИБУТОК")
    print("-" * 150)
    print(f"{'Timeframe':<12}", end="")
    for profile in profiles:
        print(f"| {profile:<15}", end="")
    print("|")
    print("-" * 150)
    
    for tf in timeframes:
        print(f"{tf:<12}", end="")
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            print(f"| {config['expected_return']:<15}", end="")
        print("|")
    
    print("\n" + "=" * 150)


def print_detailed_comparison():
    """Вивести детальне порівняння"""
    
    print("\n" + "=" * 100)
    print("📋 ДЕТАЛЬНЕ ПОРІВНЯННЯ ПО ТАЙМФРЕЙМАМ")
    print("=" * 100)
    
    timeframes = TimeframeParameters.get_all_timeframes()
    
    for tf in timeframes:
        config_balanced = TimeframeParameters.get_timeframe_config(tf, 'BALANCED')
        
        print(f"\n{'─' * 100}")
        print(f"📊 {tf.upper():>6} - {config_balanced['strategy_type']:<12} | {config_balanced['hold_time']}")
        print(f"{'─' * 100}")
        
        print(f"  RSI Period:        {config_balanced['rsi_period']:>3}  |  "
              f"RSI Threshold:  {config_balanced['rsi_threshold_long']:>3} / {config_balanced['rsi_threshold_short']:<3}  |  "
              f"Trades/day: {config_balanced['trades_per_day']:<8}")
        
        print(f"  Trailing Stop:  {config_balanced['trailing_stop_percent']*100:>6.2f}%  |  "
              f"Min Divergence: {config_balanced['min_divergence_candles']:>2}  |  "
              f"Expected: {config_balanced['expected_return']:<15}")
        
        print(f"  SL: {config_balanced['stop_loss_percent']:>5.2f}%  |  "
              f"TP: {config_balanced['take_profit_percent']:>5.2f}%  |  "
              f"Max Loss: {config_balanced['expected_max_loss']:<8}")


def print_profile_comparison():
    """Порівняння профілів на 1h"""
    
    print("\n" + "=" * 100)
    print("🎯 ПОРІВНЯННЯ ПРОФІЛІВ НА 1H ТАЙМФРЕЙМУ")
    print("=" * 100)
    
    profiles = TimeframeParameters.get_all_profiles()
    
    print(f"\n{'Profile':<15} | {'RSI TH':<8} | {'TS%':<8} | {'DIV':<4} | {'SL%':<7} | {'TP%':<7} | {'Expected':<15}")
    print("─" * 100)
    
    for profile in profiles:
        config = TimeframeParameters.get_timeframe_config('1h', profile)
        
        print(f"{profile:<15} | "
              f"{config['rsi_threshold_long']:<8} | "
              f"{config['trailing_stop_percent']*100:>6.2f}% | "
              f"{config['min_divergence_candles']:<4} | "
              f"{config['stop_loss_percent']:>5.2f}% | "
              f"{config['take_profit_percent']:>5.2f}% | "
              f"{config['expected_return']:<15}")


def generate_html_table():
    """Генерувати HTML таблицю для веб-інтерфейсу"""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Trading Bot Parameters</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #4CAF50; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .header { font-size: 24px; font-weight: bold; margin: 20px 0; }
        </style>
    </head>
    <body>
    """
    
    timeframes = TimeframeParameters.get_all_timeframes()
    profiles = TimeframeParameters.get_all_profiles()
    
    # RSI Threshold таблиця
    html += "<div class='header'>📊 RSI THRESHOLD (для LONG)</div>\n"
    html += "<table>\n<tr><th>Timeframe</th>"
    for profile in profiles:
        html += f"<th>{profile}</th>"
    html += "</tr>\n"
    
    for tf in timeframes:
        html += f"<tr><td><strong>{tf}</strong></td>"
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            html += f"<td>{config['rsi_threshold_long']}</td>"
        html += "</tr>\n"
    
    html += "</table>\n"
    
    # Trailing Stop таблиця
    html += "<div class='header'>🎯 TRAILING STOP (%)</div>\n"
    html += "<table>\n<tr><th>Timeframe</th>"
    for profile in profiles:
        html += f"<th>{profile}</th>"
    html += "</tr>\n"
    
    for tf in timeframes:
        html += f"<tr><td><strong>{tf}</strong></td>"
        for profile in profiles:
            config = TimeframeParameters.get_timeframe_config(tf, profile)
            ts = config['trailing_stop_percent'] * 100
            html += f"<td>{ts:.2f}%</td>"
        html += "</tr>\n"
    
    html += "</table>\n"
    
    html += """
    </body>
    </html>
    """
    
    return html


if __name__ == "__main__":
    
    print("\n" + "🎯" * 50)
    print("📊 ПАРАМЕТРИ ДЛЯ ВСІХ ТАЙМФРЕЙМІВ ТА ПРОФІЛІВ")
    print("🎯" * 50)
    
    # Вивести основну таблицю
    print_parameters_table()
    
    # Вивести детальне порівняння
    print_detailed_comparison()
    
    # Вивести порівняння профілів
    print_profile_comparison()
    
    # Генерувати HTML
    html_content = generate_html_table()
    with open('parameters_table.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("\n✅ HTML таблиця збережена в parameters_table.html")
    
    print("\n" + "=" * 100)
    print("✅ ВСІ ТАБЛИЦІ ВИВЕДЕНІ УСПІШНО!")
    print("=" * 100)
