"""
Position Monitor Scanner - Моніторинг ТІЛЬКИ активних угод
"""

import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.running = True
        
        # Зберігаємо дані тільки по активним монетам
        self.active_coins_data = {}  # { 'BTCUSDT': { 'pressure': 0, 'rsi': 50, 'history': [...] } }
        self.last_scan_time = None
        
        # Скануємо частіше, бо монет мало (наприклад, кожні 5-10 сек)
        self.scan_interval = 10 

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🚀 Position Monitor Started (Tracking active trades only)")
    
    def loop(self):
        while self.running:
            try:
                self.monitor_positions()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            time.sleep(self.scan_interval)
    
    def get_active_symbols(self):
        """Отримати список монет, де у нас є позиція"""
        symbols = []
        try:
            # Отримуємо позиції з Bybit
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        symbols.append(p['symbol'])
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
        return symbols

    def calculate_rsi(self, symbol, current_price):
        """Рахуємо RSI локально для активної монети"""
        if symbol not in self.active_coins_data:
            self.active_coins_data[symbol] = {'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': 0}
        
        data = self.active_coins_data[symbol]
        data['prices'].append(current_price)
        
        # Тримаємо історію короткою (20 точок)
        if len(data['prices']) > 20: data['prices'].pop(0)
        
        if len(data['prices']) < 2: return 50
        
        gains = 0
        losses = 0
        for i in range(1, len(data['prices'])):
            diff = data['prices'][i] - data['prices'][i-1]
            if diff > 0: gains += diff
            else: losses -= diff
            
        if losses == 0: return 100
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 1)

    def monitor_positions(self):
        # 1. Які монети зараз в роботі?
        target_symbols = self.get_active_symbols()
        
        # Якщо угод немає - очищуємо пам'ять і чекаємо
        if not target_symbols:
            self.active_coins_data = {}
            self.last_scan_time = datetime.now()
            return

        # 2. Отримуємо дані тільки по цим монетам
        try:
            # Отримуємо тікери для всіх, але фільтруємо потрібні
            # (API Bybit не дозволяє отримати тікер для 1 монети пакетно, простіше взяти все і відфільтрувати)
            all_tickers = self.bot.get_all_tickers()
            
            for t in all_tickers:
                symbol = t['symbol']
                if symbol not in target_symbols: continue
                
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h']) # Це накопичувальний об'єм за добу
                
                # Ініціалізація структури, якщо нова монета
                if symbol not in self.active_coins_data:
                    self.active_coins_data[symbol] = {
                        'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': turnover, 'prev_price': price
                    }
                
                coin_data = self.active_coins_data[symbol]
                
                # --- РОЗРАХУНОК RSI ---
                coin_data['rsi'] = self.calculate_rsi(symbol, price)
                
                # --- РОЗРАХУНОК ТИСКУ (VOLUME PRESSURE) ---
                # Тиск = Зміна об'єму * Напрямок ціни
                # Якщо це перше сканування, pressure буде 0
                if coin_data['last_turnover'] > 0:
                    vol_diff = turnover - coin_data['last_turnover']
                    # Фільтруємо помилки API (іноді turnover скидається)
                    if vol_diff < 0: vol_diff = 0 
                    
                    price_direction = 1 if price >= coin_data.get('prev_price', price) else -1
                    
                    # Накопичуємо тиск (згасання з часом, щоб старі дані не впливали вічно)
                    # pressure * 0.9 означає, що старий тиск втрачає силу
                    current_flow = vol_diff * price_direction
                    coin_data['pressure'] = (coin_data['pressure'] * 0.8) + current_flow
                
                # Оновлюємо попередні значення
                coin_data['last_turnover'] = turnover
                coin_data['prev_price'] = price
                
                logger.info(f"📊 {symbol}: RSI={coin_data['rsi']}, Pressure=${int(coin_data['pressure'])}")

        except Exception as e:
            logger.error(f"Scan error: {e}")
            
        self.last_scan_time = datetime.now()

    # === МЕТОДИ ДЛЯ ЗОВНІШНЬОГО ДОСТУПУ ===
    
    def get_current_rsi(self, symbol):
        """Повертає RSI для активної монети (або 50, якщо немає даних)"""
        return self.active_coins_data.get(symbol, {}).get('rsi', 50)

    def get_market_pressure(self, symbol):
        """Повертає Тиск Об'єму для активної монети"""
        return self.active_coins_data.get(symbol, {}).get('pressure', 0)

    def get_aggregated_data(self, hours=24):
        """Для сумісності з UI (повертає пустий список сигналів, бо ми не шукаємо нові)"""
        return {
            'all_signals': [], # Сканер тепер не генерує сигнали на вхід
            'last_scan': self.last_scan_time,
            'snapshots': self.active_coins_data # Передаємо дані активних монет
        }
