"""
Scanner Module - Active Position Monitor Only 🎯
Цей модуль більше не шукає нові сигнали.
Він моніторить ТІЛЬКИ відкриті угоди, щоб надавати дані (RSI, Тиск) для UI.
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
        
        # Тут зберігаємо дані тільки по активним монетам
        # Структура: { 'BTCUSDT': { 'rsi': 55, 'pressure': 12000, ... } }
        self.active_coins_data = {} 
        self.last_scan_time = None
        
        # Скануємо часто, бо монет мало (тільки активні)
        self.scan_interval = 5 

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
            # Запитуємо у біржі, що ми зараз тримаємо
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        symbols.append(p['symbol'])
        except: pass
        return symbols

    def calculate_rsi(self, symbol, current_price):
        """Локальний розрахунок RSI для активної монети"""
        if symbol not in self.active_coins_data:
            self.active_coins_data[symbol] = {'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': 0}
        
        data = self.active_coins_data[symbol]
        data['prices'].append(current_price)
        
        # Зберігаємо історію (30 точок достатньо для динаміки)
        if len(data['prices']) > 30: data['prices'].pop(0)
        
        # Якщо даних мало - повертаємо нейтральне значення
        if len(data['prices']) < 5: return 50
        
        gains = 0
        losses = 0
        for i in range(1, len(data['prices'])):
            diff = data['prices'][i] - data['prices'][i-1]
            if diff > 0: gains += diff
            else: losses -= diff
            
        if losses == 0: return 99 if gains > 0 else 50
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
            all_tickers = self.bot.get_all_tickers()
            
            for t in all_tickers:
                symbol = t['symbol']
                # Ігноруємо все, що не в роботі
                if symbol not in target_symbols: continue
                
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h']) 
                
                # Ініціалізація
                if symbol not in self.active_coins_data:
                    self.active_coins_data[symbol] = {
                        'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': turnover, 'prev_price': price
                    }
                
                coin_data = self.active_coins_data[symbol]
                
                # --- РОЗРАХУНОК RSI ---
                coin_data['rsi'] = self.calculate_rsi(symbol, price)
                
                # --- РОЗРАХУНОК ТИСКУ (VOLUME PRESSURE) ---
                # Тиск = (Зміна об'єму) * (Напрямок ціни)
                if coin_data['last_turnover'] > 0:
                    vol_diff = turnover - coin_data['last_turnover']
                    if vol_diff < 0: vol_diff = 0 # Фільтр глюків API
                    
                    price_dir = 1 if price >= coin_data.get('prev_price', price) else -1
                    
                    # Накопичувальний тиск з коефіцієнтом згасання (0.9)
                    # Це показує імпульс саме за останні хвилини
                    current_flow = vol_diff * price_dir
                    coin_data['pressure'] = (coin_data['pressure'] * 0.9) + current_flow
                
                # Оновлюємо попередні значення
                coin_data['last_turnover'] = turnover
                coin_data['prev_price'] = price
                
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            
        self.last_scan_time = datetime.now()

    # === МЕТОДИ ДЛЯ UI (main_app.py) ===
    
    def get_current_rsi(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('rsi', 50)

    def get_market_pressure(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('pressure', 0)

    def get_aggregated_data(self, hours=24):
        # Повертаємо пустий all_signals, бо ми не шукаємо нові
        return {
            'all_signals': [], 
            'last_scan': self.last_scan_time,
            'snapshots': self.active_coins_data
        }
