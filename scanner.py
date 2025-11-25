# scanner.py
import threading
import time
import logging
from datetime import datetime, timedelta
from statistics_service import stats_service

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.running = True
        self.previous_snapshot = {} # Тут зберігається поточний RSI
        self.detected_pumps = [] 
        self.last_scan_time = None
        self.rsi_cache = {} 
        
        self.scan_interval = config.get('SCANNER_INTERVAL', 60)
        self.volume_spike_threshold = config.get('VOLUME_SPIKE_THRESHOLD', 2.0)
        self.min_inflow_value = config.get('MIN_INFLOW_VALUE', 1000000)
        self.min_24h_volume = config.get('MIN_24H_VOLUME', 10000000)
    
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🚀 Scanner Started")
    
    def loop(self):
        while self.running:
            try: self.scan_market()
            except Exception as e: logger.error(f"Scanner loop error: {e}")
            time.sleep(self.scan_interval)
    
    def calculate_rsi(self, symbol, current_price):
        if symbol not in self.rsi_cache:
            self.rsi_cache[symbol] = {'prices': [], 'rsi': 50}
        
        history = self.rsi_cache[symbol]
        history['prices'].append(current_price)
        if len(history['prices']) > 20: history['prices'].pop(0) # Тримаємо 20 останніх точок
        
        if len(history['prices']) < 2: return 50
        
        gains = 0
        losses = 0
        for i in range(1, len(history['prices'])):
            diff = history['prices'][i] - history['prices'][i-1]
            if diff > 0: gains += diff
            else: losses -= diff
            
        if losses == 0: return 100
        rs = gains / losses
        return round(100 - (100 / (1 + rs)), 1)

    # ✅ НОВИЙ МЕТОД: Щоб бот міг запитати RSI конкретної монети
    def get_current_rsi(self, symbol):
        if symbol in self.previous_snapshot:
            return self.previous_snapshot[symbol].get('rsi', 50)
        return 50

    def scan_market(self):
        tickers = self.bot.get_all_tickers()
        current_snapshot = {}
        now = datetime.now()
        
        for t in tickers:
            symbol = t['symbol']
            if "USDT" not in symbol: continue
            try:
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h'])
                rsi = self.calculate_rsi(symbol, price)
                
                if turnover < self.min_24h_volume: continue
                
                current_snapshot[symbol] = {
                    'price': price, 'turnover': turnover, 'timestamp': time.time(),
                    'change24h': float(t['price24hPcnt']) * 100, 'rsi': rsi
                }
            except: continue
        
        if self.previous_snapshot:
            self.analyze_changes(current_snapshot, now)
            
        self.previous_snapshot = current_snapshot
        self.last_scan_time = now
    
    def analyze_changes(self, current, now):
        detected = []
        for symbol, data in current.items():
            if symbol not in self.previous_snapshot: continue
            prev = self.previous_snapshot[symbol]
            
            vol_diff = data['turnover'] - prev['turnover']
            time_delta = (data['timestamp'] - prev['timestamp']) / 60
            if time_delta == 0: continue
            
            avg_vol = data['turnover'] / 1440
            expected_vol = avg_vol * time_delta
            spike = vol_diff / expected_vol if expected_vol > 0 else 0
            
            if vol_diff > self.min_inflow_value and spike > self.volume_spike_threshold:
                signal = {
                    "symbol": symbol, "price": data['price'], "spike_factor": round(spike, 1),
                    "vol_inflow": round(vol_diff, 0), 
                    "price_change_interval": round(((data['price']-prev['price'])/prev['price'])*100, 2),
                    "rsi": data['rsi'], "time": now.strftime('%H:%M:%S'), "timestamp_dt": now
                }
                detected.append(signal)
                stats_service.save_whale_signal(signal)

        if detected:
            self.detected_pumps = detected + self.detected_pumps
            
        one_day_ago = now - timedelta(hours=24)
        self.detected_pumps = [p for p in self.detected_pumps if p.get('timestamp_dt', now) > one_day_ago]

    def get_aggregated_data(self, hours=24):
        return {'all_signals': self.detected_pumps, 'last_scan': self.last_scan_time}
