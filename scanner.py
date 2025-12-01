"""
Scanner Module - Active Position Monitor
Updated: Added Rate Limit protection.
"""

import threading
import time
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from settings_manager import settings

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.running = True
        
        # Кеш даних
        self.active_coins_data = {} 
        self.last_scan_time = None
        
        # Інтервал оновлення (секунд)
        self.scan_interval = 5 

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🚀 Position Monitor Started (Syncing with Strategy TF)")
    
    def loop(self):
        while self.running:
            try:
                self.monitor_positions()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            time.sleep(self.scan_interval)
    
    def get_active_symbols(self):
        """Отримати список монет, де є відкрита позиція"""
        symbols = []
        try:
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        symbols.append(p['symbol'])
        except Exception as e:
            logger.error(f"Active symbols error: {e}")
        return symbols

    def fetch_rsi_from_api(self, symbol):
        """
        Завантажує свічки з Bybit та рахує точний RSI
        """
        try:
            tf = settings.get("ltfSelection")
            if not tf: tf = "15"
            
            resp = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=str(tf),
                limit=30
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                data = resp['result']['list']
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'turn'])
                df['close'] = df['close'].astype(float)
                df = df.iloc[::-1] 
                
                rsi_series = ta.rsi(df['close'], length=14)
                if rsi_series is not None and not rsi_series.empty:
                    return round(rsi_series.iloc[-1], 1)
                    
        except Exception as e:
            pass
            
        return 50.0

    def monitor_positions(self):
        target_symbols = self.get_active_symbols()
        
        current_keys = list(self.active_coins_data.keys())
        for k in current_keys:
            if k not in target_symbols:
                del self.active_coins_data[k]

        if not target_symbols:
            return

        try:
            all_tickers = self.bot.get_all_tickers()
            
            for t in all_tickers:
                symbol = t['symbol']
                if symbol not in target_symbols: continue
                
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h']) 
                
                if symbol not in self.active_coins_data:
                    self.active_coins_data[symbol] = {
                        'rsi': 50, 'pressure': 0, 'last_turnover': turnover, 'prev_price': price
                    }
                
                coin_data = self.active_coins_data[symbol]
                
                # --- РОЗРАХУНОК RSI ---
                coin_data['rsi'] = self.fetch_rsi_from_api(symbol)
                
                # Пауза між запитами RSI для кожної активної монети, щоб не перевищити ліміт
                time.sleep(0.2) 
                
                # --- РОЗРАХУНОК ТИСКУ ---
                if coin_data['last_turnover'] > 0:
                    vol_diff = turnover - coin_data['last_turnover']
                    if vol_diff < 0: vol_diff = 0
                    
                    price_dir = 1 if price >= coin_data.get('prev_price', price) else -1
                    current_flow = vol_diff * price_dir
                    coin_data['pressure'] = (coin_data['pressure'] * 0.9) + current_flow
                
                coin_data['last_turnover'] = turnover
                coin_data['prev_price'] = price
                
        except Exception as e:
            logger.error(f"Monitor calculation error: {e}")

    # === МЕТОДИ ДЛЯ UI ===
    
    def get_current_rsi(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('rsi', 50)

    def get_market_pressure(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('pressure', 0)