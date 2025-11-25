"""
Enhanced Market Scanner з інтеграцією бази даних
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from statistics_service import stats_service

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    """Покращений сканер ринку з персистентністю даних"""
    
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.running = True
        self.previous_snapshot = {}
        self.last_scan_time = None
        
        # Параметри з конфігу
        self.scan_interval = config.get('SCANNER_INTERVAL', 60)
        self.volume_spike_threshold = config.get('VOLUME_SPIKE_THRESHOLD', 2.0)
        self.min_inflow_value = config.get('MIN_INFLOW_VALUE', 1000000)
        self.min_24h_volume = config.get('MIN_24H_VOLUME', 10000000)
    
    def start(self):
        """Запустити сканер в окремому потоці"""
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🚀 Enhanced Market Scanner Started")
    
    def loop(self):
        """Основний цикл сканування"""
        while self.running:
            try:
                self.scan_market()
            except Exception as e:
                logger.error(f"❌ Scanner error: {e}")
            time.sleep(self.scan_interval)
    
    def scan_market(self):
        """Сканувати ринок та зберегти результати в БД"""
        tickers = self.bot.get_all_tickers()
        current_snapshot = {}
        
        # 1. Збір поточних даних
        for t in tickers:
            symbol = t['symbol']
            if "USDT" not in symbol:
                continue
            
            try:
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h'])
                
                # Фільтр ліквідності
                if turnover < self.min_24h_volume:
                    continue
                
                current_snapshot[symbol] = {
                    'price': price,
                    'turnover': turnover,
                    'timestamp': time.time(),
                    'change24h': float(t['price24hPcnt']) * 100
                }
            except Exception as e:
                continue
        
        # 2. Аналіз змін (якщо є попередні дані)
        if self.previous_snapshot:
            detected_signals = self.analyze_changes(current_snapshot)
            
            # 3. Зберегти сигнали в БД
            for signal in detected_signals:
                stats_service.save_whale_signal(signal)
        
        self.previous_snapshot = current_snapshot
        self.last_scan_time = datetime.now()
    
    def analyze_changes(self, current):
        """Аналіз змін та виявлення аномалій"""
        detected = []
        now = datetime.now()
        
        for symbol, data in current.items():
            if symbol not in self.previous_snapshot:
                continue
            
            prev = self.previous_snapshot[symbol]
            time_delta = (data['timestamp'] - prev['timestamp']) / 60  # хвилини
            
            if time_delta == 0:
                continue
            
            # Розрахунок вливання
            vol_diff = data['turnover'] - prev['turnover']
            
            if vol_diff < self.min_inflow_value:
                continue
            
            # Очікуваний об'єм
            avg_vol_per_min = data['turnover'] / 1440  # За добу
            expected_vol = avg_vol_per_min * time_delta
            
            # Фактор аномалії
            spike_factor = vol_diff / expected_vol if expected_vol > 0 else 0
            
            # Зміна ціни за інтервал
            price_change_interval = ((data['price'] - prev['price']) / prev['price']) * 100
            
            # Перевірка порогу
            if spike_factor > self.volume_spike_threshold:
                signal = {
                    "symbol": symbol,
                    "price": data['price'],
                    "spike_factor": round(spike_factor, 1),
                    "vol_inflow": round(vol_diff, 0),
                    "price_change_interval": round(price_change_interval, 2),
                    "turnover_24h": data['turnover'],
                    "time": now.strftime('%H:%M:%S'),
                    "timestamp_dt": now
                }
                detected.append(signal)
        
        if detected:
            logger.info(f"🚨 DETECTED {len(detected)} WHALE MOVES!")
        
        return detected
    
    def get_recent_signals(self, hours=24):
        """Отримати останні сигнали з БД"""
        return stats_service.get_whale_signals(hours=hours)
    
    def get_aggregated_data(self, hours=24):
        """
        Отримати агреговані дані для дашборду
        Замість зберігання в пам'яті - запитуємо з БД
        """
        signals = self.get_recent_signals(hours=hours)
        
        # Агрегація по монетам
        coin_stats = {}
        
        for signal in signals:
            sym = signal['symbol']
            val = signal['vol_inflow']
            change = signal['price_change_interval']
            
            if sym not in coin_stats:
                coin_stats[sym] = {
                    'inflow': 0,
                    'total_change': 0,
                    'count': 0,
                    'signals': []
                }
            
            coin_stats[sym]['inflow'] += val
            coin_stats[sym]['total_change'] += change
            coin_stats[sym]['count'] += 1
            coin_stats[sym]['signals'].append(signal)
        
        # Розділити на позитивні та негативні
        positive_coins = []
        negative_coins = []
        
        max_inflow = max((s['inflow'] for s in coin_stats.values()), default=1)
        
        for sym, data in coin_stats.items():
            avg_change = data['total_change'] / data['count']
            bar_pct = (data['inflow'] / max_inflow) * 100 if max_inflow > 0 else 0
            
            coin_data = {
                'symbol': sym,
                'inflow': data['inflow'],
                'avg_change': round(avg_change, 2),
                'count': data['count'],
                'bar_pct': bar_pct,
                'last_signal': data['signals'][-1] if data['signals'] else None
            }
            
            if avg_change >= 0:
                positive_coins.append(coin_data)
            else:
                negative_coins.append(coin_data)
        
        # Сортування
        positive_coins.sort(key=lambda x: x['inflow'], reverse=True)
        negative_coins.sort(key=lambda x: x['inflow'], reverse=True)
        
        return {
            'positive_coins': positive_coins,
            'negative_coins': negative_coins,
            'all_signals': signals,
            'last_scan': self.last_scan_time
        }
    
    def get_top_movers(self, hours=24, limit=10):
        """Топ монет за період"""
        return stats_service.get_top_coins(period_hours=hours, limit=limit, sort_by='inflow')
    
    def stop(self):
        """Зупинити сканер"""
        self.running = False
        logger.info("🛑 Scanner stopped")
