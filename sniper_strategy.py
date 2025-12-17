#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 WHALE SNIPER STRATEGY
========================
Файл: sniper_strategy.py

Гібридна стратегія: Whale Trend + Smart Money Structure + RSI Trigger.
Працює в окремому потоці, не блокує основного бота.
"""

import time
import threading
import logging
import pandas as pd
from datetime import datetime

# Імпорти з проекту
from bot import bot_instance
from settings_manager import settings
from models import db_manager, WhaleSignal
from indicators import simple_rsi

# Імпорт логіки з інших модулів (Reuse)
from whale_pro import whale_pro
from order_block_scanner import OrderBlockScanner

logger = logging.getLogger("SniperStrategy")

class WhaleSniper:
    def __init__(self):
        self.is_running = False
        self.status = "Ready"
        self.last_run_time = None
        self._stop_event = threading.Event()
        self.found_signals = [] # Локальний кеш для UI
        
        # Налаштування Снайпера
        self.CONFIG = {
            "btc_check": True,          # Тільки якщо BTC Bullish
            "min_vol_usdt": 15_000_000, # Ліквідність > 15M
            "rvol_min": 2.0,            # Сильний сплеск об'єму (Whale)
            "adx_min": 20,              # Наявність тренду
            "ob_timeframe": "15",       # Робочий ТФ для входу
            "rsi_trigger": 40,          # RSI має бути низьким при торканні OB
            "scan_interval": 300        # Пауза 5 хв між циклами
        }
        
        # Ініціалізація сканера OB з агресивними налаштуваннями
        ob_settings = {
            'ob_source_tf': self.CONFIG['ob_timeframe'],
            'ob_swing_length': 5,
            'ob_zone_count': 'High',
            'ob_max_atr_mult': 4.0,     
            'ob_combine_obs': True,
            'ob_entry_mode': 'Immediate', 
            'ob_selection': 'Closest'
        }
        # Створюємо екземпляр сканера
        self.ob_scanner = OrderBlockScanner(bot_instance.session, ob_settings)

    def check_market_conditions(self):
        """КРОК 1: Перевірка BTC"""
        if not self.CONFIG['btc_check']:
            return True
            
        trend = whale_pro.analyze_btc_trend()
        if trend != "BULLISH":
            self.status = f"Paused: BTC is {trend}"
            return False
        return True

    def find_whale_tracks(self):
        """КРОК 2: Пошук монет з RVOL > 2"""
        self.status = "Scanning Vol & Trend..."
        tickers = bot_instance.get_all_tickers()
        candidates = []
        
        # Фільтр ліквідності
        targets = [
            t for t in tickers 
            if t['symbol'].endswith('USDT') 
            and float(t.get('turnover24h', 0)) > self.CONFIG['min_vol_usdt']
        ]
        
        # Сортуємо по об'єму, беремо ТОП-40
        targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        check_list = targets[:40]
        
        for t in check_list:
            if not self.is_running: break
            
            sym = t['symbol']
            # Використовуємо функцію з whale_pro, яка вже має limit=1000
            df = whale_pro.fetch_data(sym)
            
            if df is not None:
                res = whale_pro.analyze_ticker_pro(sym, df)
                if res:
                    # Фільтр: Є об'єм і є хоч якийсь тренд
                    if res['rvol'] >= self.CONFIG['rvol_min'] and res['adx'] >= self.CONFIG['adx_min']:
                        candidates.append(sym)
                        
            time.sleep(0.1) # Захист API
            
        return candidates

    def hunt_targets(self, candidates):
        """КРОК 3 і 4: Перевірка OB та RSI"""
        self.status = f"Hunting in {len(candidates)} coins..."
        
        for sym in candidates:
            if not self.is_running: break
            
            # 1. Шукаємо Order Block (Тільки BUY)
            # Сканер сам завантажить 1000 свічок завдяки нашим правкам
            ob_result = self.ob_scanner.scan_symbol(sym, "BUY")
            
            if ob_result and ob_result.get('status') in ['Valid', 'Triggered']:
                entry = ob_result['entry_price']
                curr = ob_result['current_price']
                
                # Відстань до входу у відсотках
                dist = (curr - entry) / entry * 100
                
                # ЛОГІКА: Ціна має бути або в зоні, або трохи вище (до 1%)
                # Ми чекаємо відкату
                if -1.0 <= dist <= 1.0:
                    
                    # 2. RSI Trigger
                    # Довантажуємо дані для точного RSI
                    df = whale_pro.fetch_data(sym)
                    if df is not None:
                        rsi = simple_rsi(df['close'], period=14).iloc[-1]
                        
                        # Якщо RSI перепроданий - це СИГНАЛ
                        if rsi <= self.CONFIG['rsi_trigger']:
                            self.fire_signal(sym, curr, rsi, ob_result, dist)
                            
            time.sleep(0.1)

    def fire_signal(self, symbol, price, rsi, ob_data, dist):
        """Збереження сигналу"""
        log_msg = f"🎯 SNIPER: {symbol} | Price: {price} | RSI: {round(rsi,1)} | Near OB ({round(dist,2)}%)"
        logger.info(log_msg)
        
        session = db_manager.get_session()
        try:
            # Зберігаємо в ту ж таблицю, що і Whale, але з поміткою SNIPER
            sig = WhaleSignal(
                symbol=symbol,
                price=price,
                score=99, # Елітний сигнал
                squeeze_val=0,
                obv_slope=0,
                details=f"SNIPER 🎯 (RVOL+OB+RSI {round(rsi,1)})",
                created_at=datetime.utcnow(),
                rsi=rsi
            )
            session.add(sig)
            session.commit()
            
            # Додаємо в локальний список для UI
            self.found_signals.insert(0, {
                'time': datetime.now().strftime("%H:%M"),
                'symbol': symbol,
                'price': price,
                'rsi': round(rsi, 1),
                'ob_entry': ob_data['entry_price'],
                'details': f"Retest OB + RSI < {self.CONFIG['rsi_trigger']}"
            })
            self.found_signals = self.found_signals[:20] # Тримаємо останні 20
            
        except Exception as e:
            logger.error(f"Save sniper error: {e}")
        finally:
            session.close()

    def start(self):
        if self.is_running: return
        self.is_running = True
        self._stop_event.clear()
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Whale Sniper Started")

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        self.status = "Stopped"

    def _loop(self):
        while self.is_running:
            try:
                # 1. Перевірка BTC
                if self.check_market_conditions():
                    
                    # 2. Пошук китів
                    candidates = self.find_whale_tracks()
                    
                    if candidates:
                        # 3. Полювання
                        self.hunt_targets(candidates)
                    else:
                        self.status = "No Whale Volume found"
                        
                    self.last_run_time = datetime.now().strftime("%H:%M")
                    self.status = "Waiting next cycle..."
                
                # Очікування з можливістю зупинки
                for _ in range(self.CONFIG['scan_interval']):
                    if not self.is_running: break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Sniper loop crash: {e}")
                self.status = "Error (Restarting...)"
                time.sleep(60)

# Глобальний екземпляр
sniper_bot = WhaleSniper()
