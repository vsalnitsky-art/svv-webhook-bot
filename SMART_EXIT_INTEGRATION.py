#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 ІНСТРУКЦІЯ ІНТЕГРАЦІЇ SMART EXIT В SCANNER.PY

Файл: scanner.py
Метод: monitor_positions()

Замніть поточну логіку на нову.
"""

# ====================================================================
# КРОК 1: ІМПОРТ У ПОЧАТКУ ФАЙЛУ scanner.py
# ====================================================================

from smart_exit_strategy import smart_exit


# ====================================================================
# КРОК 2: ЗАМІНА МЕТОДУ monitor_positions()
# ====================================================================

def monitor_positions(self):
    """
    ✅ НОВА ЛОГІКА: Smart Exit з Trailing Stop + Дивергенцією
    """
    
    try:
        # Отримуємо активні позиції
        active_positions = self.get_active_symbols()
        
        if not active_positions:
            return
        
        for pos in active_positions:
            symbol = pos['symbol']
            side = pos['side']  # 'Buy' або 'Sell'
            
            # ✅ КРОК 1: Завантажуємо HTF свічки
            df_htf = self.fetch_htf_candles(symbol)
            
            if df_htf is None or len(df_htf) < 2:
                continue
            
            # ✅ КРОК 2: Отримуємо поточну ціну та RSI
            current_price = df_htf['close'].iloc[-1]
            rsi_value = df_htf['rsi'].iloc[-1]
            
            # ✅ КРОК 3: Оновлюємо позицію в Smart Exit
            exit_signal = smart_exit.update_position(
                symbol=symbol,
                current_price=current_price,
                rsi_value=rsi_value,
                side='Long' if side == 'Buy' else 'Short'
            )
            
            # ✅ КРОК 4: Перевіряємо сигнал на закриття
            if exit_signal['should_close']:
                logger.warning(
                    f"🚨 SMART EXIT SIGNAL: {symbol}\n"
                    f"   Reason: {exit_signal['reason']}\n"
                    f"   Exit Price: {exit_signal['exit_price']}\n"
                    f"   Profit: {exit_signal['profit_potential']:.2f}%"
                )
                
                # Закриваємо позицію
                result = self.bot.place_order({
                    "action": "Close",
                    "symbol": symbol,
                    "direction": "Long" if side == "Buy" else "Short"
                })
                
                if result.get("status") == "ok":
                    logger.info(f"✅ Position closed: {symbol}")
                    # Скидаємо стан позиції
                    smart_exit.reset_position(symbol)
            
            else:
                # Для дебагу: виводимо стан
                status = smart_exit.get_position_status(symbol)
                if status['status'] != 'WAITING for RSI >= 70':
                    logger.debug(f"📊 {symbol}: {status['status']}")
    
    except Exception as e:
        logger.error(f"❌ Error in monitor_positions: {e}")


# ====================================================================
# КРОК 3: НАЛАШТУВАННЯ (за бажанням)
# ====================================================================

# Якщо хочете змінити параметри, додайте перед вживанням:

def configure_smart_exit():
    """Налаштування Smart Exit"""
    
    from smart_exit_strategy import smart_exit
    
    # Змініть ці значення за потребою:
    smart_exit.TRAILING_STOP_PERCENT = -0.005      # -0.5% (змініть на -0.003 = -0.3%)
    smart_exit.MIN_DIVERGENCE_CANDLES = 2          # Мінімум 2 свічки
    smart_exit.RSI_THRESHOLD = 70                  # Для LONG (для SHORT = 30)
    
    logger.info(f"✅ Smart Exit configured:")
    logger.info(f"   Trailing Stop: {smart_exit.TRAILING_STOP_PERCENT * 100}%")
    logger.info(f"   Min Divergence Candles: {smart_exit.MIN_DIVERGENCE_CANDLES}")
    logger.info(f"   RSI Threshold: {smart_exit.RSI_THRESHOLD}")


# Викличте при запуску бота:
# configure_smart_exit()


# ====================================================================
# КРОК 4: ЛОГУВАННЯ ДЛЯ DEBUGGING
# ====================================================================

# Додайте цей метод в scanner.py:

def print_smart_exit_status(self):
    """Вивести статус всіх позицій"""
    
    active_symbols = [p['symbol'] for p in self.get_active_symbols()]
    
    logger.info("=" * 60)
    logger.info("SMART EXIT STATUS")
    logger.info("=" * 60)
    
    for symbol in active_symbols:
        status = smart_exit.get_position_status(symbol)
        
        if status['status'] == 'HUNTING maximum':
            details = status['details']
            logger.info(
                f"\n🎯 {symbol}:\n"
                f"   Status: {status['status']}\n"
                f"   Entry: ${details['entry_price']:.2f}\n"
                f"   Fresh Max: ${details['fresh_max_price']:.2f}\n"
                f"   Trailing Stop: ${details['trailing_stop']:.2f}\n"
                f"   Divergence Count: {details['divergence_count']}\n"
                f"   Time Active: {details['time_active']:.0f}s"
            )
        else:
            logger.info(f"\n⏳ {symbol}: {status['status']}")
    
    logger.info("=" * 60)


# ====================================================================
# КРОК 5: ТЕСТУВАННЯ
# ====================================================================

# Для локального тестування:

if __name__ == "__main__":
    from smart_exit_strategy import smart_exit
    
    # Симуляція LONG позиції BTCUSDT
    print("🧪 ТЕСТ 1: Trailing Stop Hit")
    print("-" * 60)
    
    symbol = "BTCUSDT"
    entry = 50000
    
    # Фаза 1: WAITING
    print(f"1. Price: {entry}, RSI: 65")
    result = smart_exit.update_position(symbol, entry, 65, 'Long')
    print(f"   Result: {result['reason']}\n")
    
    # Фаза 2: HUNTING активована
    print(f"2. Price: {entry + 500}, RSI: 71 ⚡")
    result = smart_exit.update_position(symbol, entry + 500, 71, 'Long')
    print(f"   Result: {result['reason']}")
    status = smart_exit.get_position_status(symbol)
    print(f"   Status: {status['status']}\n")
    
    # Фаза 3: Ловимо максимум
    print(f"3. Price: {entry + 1000}, RSI: 74")
    result = smart_exit.update_position(symbol, entry + 1000, 74, 'Long')
    print(f"   Result: {result['reason']}\n")
    
    # Фаза 4: Trailing Stop hit!
    print(f"4. Price: {entry + 495}, RSI: 72 🛑")
    result = smart_exit.update_position(symbol, entry + 495, 72, 'Long')
    print(f"   Result: {result['reason']}")
    print(f"   Should Close: {result['should_close']}")
    print(f"   Reason: {result['reason']}")
    print(f"   Profit: {result['profit_potential']:.2f}%\n")
    
    # ====================================================================
    
    print("\n🧪 ТЕСТ 2: Дивергенція")
    print("-" * 60)
    
    symbol2 = "ETHUSDT"
    entry2 = 2000
    
    smart_exit.reset_position(symbol2)
    
    # Активація
    print(f"1. Price: {entry2 + 100}, RSI: 72 ⚡")
    smart_exit.update_position(symbol2, entry2 + 100, 72, 'Long')
    
    # Нові максимуми
    print(f"2. Price: {entry2 + 200}, RSI: 75")
    smart_exit.update_position(symbol2, entry2 + 200, 75, 'Long')
    
    # Дивергенція 1
    print(f"3. Price: {entry2 + 250}, RSI: 73 ⚠️ (дивергенція 1)")
    result = smart_exit.update_position(symbol2, entry2 + 250, 73, 'Long')
    print(f"   Should Close: {result['should_close']}\n")
    
    # Дивергенція 2
    print(f"4. Price: {entry2 + 260}, RSI: 70 ⚠️ (дивергенція 2)")
    result = smart_exit.update_position(symbol2, entry2 + 260, 70, 'Long')
    print(f"   Should Close: {result['should_close']}")
    print(f"   Reason: {result['reason']}")
    print(f"   Profit: {result['profit_potential']:.2f}%")
