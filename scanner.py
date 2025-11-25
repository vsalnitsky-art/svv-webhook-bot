"""
Position Monitor Scanner - Safe RSI Edition
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
        self.active_coins_data = {} 
        self.last_scan_time = None
        self.scan_interval = 5  # Частота оновлення (сек)

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🚀 Position Monitor Started")
    
    def loop(self):
        while self.running:
            try: self.monitor_positions()
            except Exception as e: logger.error(f"Monitor loop error: {e}")
            time.sleep(self.scan_interval)
    
    def get_active_symbols(self):
        symbols = []
        try:
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        symbols.append(p['symbol'])
        except: pass
        return symbols

    def calculate_rsi(self, symbol, current_price):
        """Безпечний розрахунок RSI"""
        if symbol not in self.active_coins_data:
            self.active_coins_data[symbol] = {'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': 0}
        
        data = self.active_coins_data[symbol]
        data['prices'].append(current_price)
        
        if len(data['prices']) > 30: data['prices'].pop(0)
        
        # 🔥 ЗАХИСТ: Якщо даних мало (< 10 точок), повертаємо 50 (нейтрально)
        # Це запобігає миттєвому закриттю угоди на старті
        if len(data['prices']) < 10: return 50
        
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
        target_symbols = self.get_active_symbols()
        if not target_symbols:
            self.active_coins_data = {}
            self.last_scan_time = datetime.now()
            return

        try:
            all_tickers = self.bot.get_all_tickers()
            for t in all_tickers:
                symbol = t['symbol']
                if symbol not in target_symbols: continue
                
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h'])
                
                if symbol not in self.active_coins_data:
                    self.active_coins_data[symbol] = {'prices': [], 'rsi': 50, 'pressure': 0, 'last_turnover': turnover, 'prev_price': price}
                
                coin_data = self.active_coins_data[symbol]
                coin_data['rsi'] = self.calculate_rsi(symbol, price)
                
                if coin_data['last_turnover'] > 0:
                    vol_diff = turnover - coin_data['last_turnover']
                    if vol_diff < 0: vol_diff = 0 
                    price_direction = 1 if price >= coin_data.get('prev_price', price) else -1
                    coin_data['pressure'] = (coin_data['pressure'] * 0.9) + (vol_diff * price_direction)
                
                coin_data['last_turnover'] = turnover
                coin_data['prev_price'] = price
                
        except Exception as e: logger.error(f"Scan error: {e}")
        self.last_scan_time = datetime.now()

    def get_current_rsi(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('rsi', 50)

    def get_market_pressure(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('pressure', 0)

    def get_aggregated_data(self, hours=24):
        return {'all_signals': [], 'last_scan': self.last_scan_time, 'snapshots': self.active_coins_data}
