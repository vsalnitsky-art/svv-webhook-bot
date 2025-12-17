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
        self.progress = 0  # Для прогрес-бару (0-100)
        self.last_run_time = "-"
        self._stop_event = threading.Event()
        self.found_signals = [] # Локальний кеш для UI
        
        # Завантажуємо налаштування або беремо дефолтні (Агресивні)
        self.CONFIG = {
            "btc_check": settings.get("sniper_btc_check", True),
            "min_vol_usdt": float(settings.get("sniper_min_vol", 10_000_000)), 
            "rvol_min": float(settings.get("sniper_rvol", 2.2)),
            "adx_min": int(settings.get("sniper_adx", 20)),
            "ob_timeframe": "15",
            "rsi_trigger": int(settings.get("sniper_rsi", 45)),
            "ob_swing": int(settings.get("sniper_ob_swing", 3)),
            "ob_atr": float(settings.get("sniper_ob_atr", 2.5)),
            "scan_interval": 300
        }
        
        # Ініціалізація сканера OB
        self._init_scanner()

    def _init_scanner(self):
        """Ініціалізація сканера з поточними налаштуваннями"""
        ob_settings = {
            'ob_source_tf': self.CONFIG['ob_timeframe'],
            'ob_swing_length': self.CONFIG['ob_swing'],
            'ob_zone_count': 'High',
            'ob_max_atr_mult': self.CONFIG['ob_atr'],
            'ob_combine_obs': True,
            'ob_entry_mode': 'Immediate', 
            'ob_selection': 'Closest'
        }
        self.ob_scanner = OrderBlockScanner(bot_instance.session, ob_settings)

    def update_config(self):
        """Оновлення конфігурації з UI без перезапуску"""
        try:
            self.CONFIG["btc_check"] = settings.get("sniper_btc_check", True)
            self.CONFIG["min_vol_usdt"] = float(settings.get("sniper_min_vol", 10_000_000))
            self.CONFIG["rvol_min"] = float(settings.get("sniper_rvol", 2.2))
            self.CONFIG["adx_min"] = int(settings.get("sniper_adx", 20))
            self.CONFIG["rsi_trigger"] = int(settings.get("sniper_rsi", 45))
            self.CONFIG["ob_swing"] = int(settings.get("sniper_ob_swing", 3))
            self.CONFIG["ob_atr"] = float(settings.get("sniper_ob_atr", 2.5))
            
            # Переініціалізація сканера з новими параметрами OB
            self._init_scanner()
            logger.info(f"Sniper config updated: {self.CONFIG}")
        except Exception as e:
            logger.error(f"Config update error: {e}")

    def check_market_conditions(self):
        """КРОК 1: Перевірка BTC"""
        if not self.CONFIG['btc_check']:
            return True
            
        trend = whale_pro.analyze_btc_trend()
        if trend != "BULLISH":
            self.status = f"Paused: BTC is {trend}"
            self.progress = 0
            return False
        return True

    def find_whale_tracks(self):
        """КРОК 2: Пошук монет з RVOL > X"""
        self.status = "🔍 Filtering Volume & Trend..."
        self.progress = 10
        
        tickers = bot_instance.get_all_tickers()
        candidates = []
        
        # Фільтр ліквідності
        targets = [
            t for t in tickers 
            if t['symbol'].endswith('USDT') 
            and float(t.get('turnover24h', 0)) > self.CONFIG['min_vol_usdt']
        ]
        
        # Сортуємо по об'єму, беремо ТОП-50
        targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        check_list = targets[:50]
        total_check = len(check_list)
        
        for i, t in enumerate(check_list):
            if not self.is_running: break
            
            sym = t['symbol']
            # Візуалізація процесу
            self.status = f"Scanning {sym} ({i+1}/{total_check})"
            self.progress = 10 + int((i / total_check) * 40) # 10% -> 50%
            
            df = whale_pro.fetch_data(sym)
            if df is not None:
                res = whale_pro.analyze_ticker_pro(sym, df)
                if res:
                    # Фільтр: RVOL та ADX з налаштувань
                    if res['rvol'] >= self.CONFIG['rvol_min'] and res['adx'] >= self.CONFIG['adx_min']:
                        candidates.append(sym)
                        
            time.sleep(0.05)
            
        return candidates

    def hunt_targets(self, candidates):
        """КРОК 3 і 4: Перевірка OB та RSI"""
        if not candidates:
            return

        total_cand = len(candidates)
        self.status = f"🎯 Hunting in {total_cand} targets..."
        
        for i, sym in enumerate(candidates):
            if not self.is_running: break
            
            self.progress = 50 + int((i / total_cand) * 50) # 50% -> 100%
            
            # 1. Шукаємо Order Block (Тільки BUY)
            ob_result = self.ob_scanner.scan_symbol(sym, "BUY")
            
            if ob_result and ob_result.get('status') in ['Valid', 'Triggered']:
                entry = ob_result['entry_price']
                curr = ob_result['current_price']
                
                # Відстань до входу у відсотках
                dist = (curr - entry) / entry * 100
                
                # ЛОГІКА: Ціна має бути або в зоні, або трохи вище (до 1.5%)
                if -1.0 <= dist <= 1.5:
                    
                    # 2. RSI Trigger
                    df = whale_pro.fetch_data(sym)
                    if df is not None:
                        rsi = simple_rsi(df['close'], period=14).iloc[-1]
                        
                        # Порівнюємо з налаштуванням RSI Trigger
                        if rsi <= self.CONFIG['rsi_trigger']:
                            self.fire_signal(sym, curr, rsi, ob_result, dist)
                            
            time.sleep(0.1)

    def fire_signal(self, symbol, price, rsi, ob_data, dist):
        """Збереження сигналу"""
        log_msg = f"🎯 SNIPER: {symbol} | Price: {price} | RSI: {round(rsi,1)} | Near OB ({round(dist,2)}%)"
        logger.info(log_msg)
        
        session = db_manager.get_session()
        try:
            sig = WhaleSignal(
                symbol=symbol,
                price=price,
                score=99,
                squeeze_val=0,
                obv_slope=0,
                details=f"SNIPER (RVOL {self.CONFIG['rvol_min']}+ | RSI {round(rsi,1)})",
                created_at=datetime.utcnow(),
                rsi=rsi
            )
            session.add(sig)
            session.commit()
            
            self.found_signals.insert(0, {
                'time': datetime.now().strftime("%H:%M"),
                'symbol': symbol,
                'price': price,
                'rsi': round(rsi, 1),
                'ob_entry': ob_data['entry_price'],
                'details': f"Near OB ({round(dist,2)}%) + RSI < {self.CONFIG['rsi_trigger']}"
            })
            self.found_signals = self.found_signals[:20]
            
        except Exception as e:
            logger.error(f"Save sniper error: {e}")
        finally:
            session.close()

    def start(self):
        if self.is_running: return
        self.update_config() # Оновити налаштування перед стартом
        self.is_running = True
        self._stop_event.clear()
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Whale Sniper Started")

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        self.status = "Stopped"
        self.progress = 0

    def _loop(self):
        while self.is_running:
            try:
                self.progress = 0
                # 1. Перевірка BTC
                if self.check_market_conditions():
                    
                    # 2. Пошук китів
                    candidates = self.find_whale_tracks()
                    
                    if candidates:
                        # 3. Полювання
                        self.hunt_targets(candidates)
                    else:
                        self.status = "No Whale Volume found"
                        
                    self.last_run_time = datetime.now().strftime("%H:%M:%S")
                    self.status = "Waiting next cycle..."
                    self.progress = 100
                
                # Очікування
                for i in range(self.CONFIG['scan_interval']):
                    if not self.is_running: break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Sniper loop crash: {e}")
                self.status = "Error (Restarting...)"
                time.sleep(60)

sniper_bot = WhaleSniper()
