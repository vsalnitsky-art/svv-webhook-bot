#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 WHALE SNIPER STRATEGY (ADAPTIVE: LONG & SHORT)
=================================================
Гібридна стратегія: Whale Trend + Smart Money Structure + RSI Trigger.

АВТОМАТИЧНИЙ РЕЖИМ:
1. Аналізує тренд BTC (через Whale Pro).
2. Якщо BTC BULLISH -> Режим LONG (Шукає Buy OB, RSI < Trigger).
3. Якщо BTC BEARISH -> Режим SHORT (Шукає Sell OB, RSI > 100-Trigger).

Працює в окремому потоці.
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

# Імпорт логіки з інших модулів
from whale_pro import whale_pro
from order_block_scanner import OrderBlockScanner

logger = logging.getLogger("SniperStrategy")

class WhaleSniper:
    def __init__(self):
        self.is_running = False
        self.status = "Ready"
        self.progress = 0
        self.last_run_time = "-"
        self.market_mode = "NEUTRAL" # LONG, SHORT або NEUTRAL
        self._stop_event = threading.Event()
        self.found_signals = [] # Локальний кеш для UI
        
        # Налаштування
        self.CONFIG = {
            "btc_check": settings.get("sniper_btc_check", True),
            "min_vol_usdt": float(settings.get("sniper_min_vol", 10_000_000)), 
            "rvol_min": float(settings.get("sniper_rvol", 2.2)),
            "adx_min": int(settings.get("sniper_adx", 20)),
            "ob_timeframe": "15",
            "rsi_trigger": int(settings.get("sniper_rsi", 45)), # Базовий тригер для Long
            "ob_swing": int(settings.get("sniper_ob_swing", 3)),
            "ob_atr": float(settings.get("sniper_ob_atr", 2.5)),
            "scan_interval": 300
        }
        
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
        """Оновлення налаштувань без перезапуску"""
        try:
            self.CONFIG["btc_check"] = settings.get("sniper_btc_check", True)
            self.CONFIG["min_vol_usdt"] = float(settings.get("sniper_min_vol", 10_000_000))
            self.CONFIG["rvol_min"] = float(settings.get("sniper_rvol", 2.2))
            self.CONFIG["adx_min"] = int(settings.get("sniper_adx", 20))
            self.CONFIG["rsi_trigger"] = int(settings.get("sniper_rsi", 45))
            self.CONFIG["ob_swing"] = int(settings.get("sniper_ob_swing", 3))
            self.CONFIG["ob_atr"] = float(settings.get("sniper_ob_atr", 2.5))
            
            self._init_scanner()
            logger.info(f"Sniper config updated: {self.CONFIG}")
        except Exception as e:
            logger.error(f"Config update error: {e}")

    def check_market_conditions(self):
        """КРОК 1: Визначення режиму ринку (Adaptive Logic)"""
        # Якщо фільтр вимкнено - примусово ставимо LONG (або можна Neutral)
        if not self.CONFIG['btc_check']:
            self.market_mode = "LONG"
            return True
            
        trend = whale_pro.analyze_btc_trend()
        
        if trend == "BULLISH":
            self.market_mode = "LONG"
            self.status = "Market BULLISH 🟢 Scanning for LONGS"
            return True
        elif trend == "BEARISH":
            self.market_mode = "SHORT"
            self.status = "Market BEARISH 🔴 Scanning for SHORTS"
            return True
        else:
            self.market_mode = "NEUTRAL"
            self.status = f"Market {trend} ⚪ Paused"
            self.progress = 0
            return False

    def find_whale_tracks(self):
        """КРОК 2: Пошук монет з високим RVOL"""
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
            
            # Візуалізація статусу
            icon = "🟢" if self.market_mode == "LONG" else "🔴"
            self.status = f"{icon} Scanning {sym} ({i+1}/{total_check})"
            self.progress = 10 + int((i / total_check) * 40)
            
            df = whale_pro.fetch_data(sym)
            if df is not None:
                res = whale_pro.analyze_ticker_pro(sym, df)
                if res:
                    # Фільтр RVOL важливий в обидві сторони
                    if res['rvol'] >= self.CONFIG['rvol_min'] and res['adx'] >= self.CONFIG['adx_min']:
                        candidates.append(sym)
            
            time.sleep(0.05)
            
        return candidates

    def hunt_targets(self, candidates):
        """КРОК 3 і 4: Адаптивний пошук (Buy/Sell OB + RSI)"""
        if not candidates: return

        total_cand = len(candidates)
        self.status = f"🎯 Hunting in {total_cand} targets ({self.market_mode})..."
        
        # Налаштування логіки залежно від режиму
        if self.market_mode == "LONG":
            scan_direction = "BUY"
            # Для Long: RSI має бути низьким
            rsi_condition = lambda r: r <= self.CONFIG['rsi_trigger']
            trigger_text = f"RSI < {self.CONFIG['rsi_trigger']}"
        else:
            scan_direction = "SELL"
            # Для Short: RSI має бути високим (дзеркально)
            short_trigger = 100 - self.CONFIG['rsi_trigger']
            rsi_condition = lambda r: r >= short_trigger
            trigger_text = f"RSI > {short_trigger}"
        
        for i, sym in enumerate(candidates):
            if not self.is_running: break
            
            self.progress = 50 + int((i / total_cand) * 50)
            
            # 1. Шукаємо Order Block (Buy або Sell)
            ob_result = self.ob_scanner.scan_symbol(sym, scan_direction)
            
            if ob_result and ob_result.get('status') in ['Valid', 'Triggered']:
                entry = ob_result['entry_price']
                curr = ob_result['current_price']
                
                # Відстань у % (абсолютне значення)
                dist = abs((curr - entry) / entry * 100)
                
                # Ціна має бути дуже близько до блоку (до 1.5%)
                if dist <= 1.5:
                    
                    # 2. RSI Trigger
                    df = whale_pro.fetch_data(sym)
                    if df is not None:
                        rsi = simple_rsi(df['close'], period=14).iloc[-1]
                        
                        # Перевіряємо умову RSI
                        if rsi_condition(rsi):
                            self.fire_signal(sym, curr, rsi, ob_result, dist, self.market_mode, trigger_text)
                            
            time.sleep(0.1)

    def fire_signal(self, symbol, price, rsi, ob_data, dist, mode, trigger_text):
        """Збереження сигналу"""
        icon = "🟢" if mode == "LONG" else "🔴"
        log_msg = f"{icon} SNIPER: {symbol} | Price: {price} | RSI: {round(rsi,1)} | Mode: {mode}"
        logger.info(log_msg)
        
        session = db_manager.get_session()
        try:
            sig = WhaleSignal(
                symbol=symbol,
                price=price,
                score=99,
                squeeze_val=0,
                obv_slope=0,
                details=f"{mode} SNIPER (RSI {round(rsi,1)})",
                created_at=datetime.utcnow(),
                rsi=rsi
            )
            session.add(sig)
            session.commit()
            
            # Додаємо в UI
            self.found_signals.insert(0, {
                'time': datetime.now().strftime("%H:%M"),
                'symbol': symbol,
                'price': price,
                'rsi': round(rsi, 1),
                'mode': mode,
                'ob_entry': ob_data['entry_price'],
                'details': f"Near OB ({round(dist,2)}%) | {trigger_text}"
            })
            self.found_signals = self.found_signals[:20]
            
            # 🆕 ІНТЕГРАЦІЯ: Додаємо до Smart Money Watchlist
            if settings.get('sniper_add_to_watchlist', True):
                try:
                    from scanner_coordinator import add_to_smart_money_watchlist
                    
                    direction = 'BUY' if mode == 'LONG' else 'SELL'
                    add_result = add_to_smart_money_watchlist(
                        symbol=symbol,
                        direction=direction,
                        source='Whale SNIPER'
                    )
                    
                    if add_result.get('status') == 'ok':
                        logger.info(f"📋 Added to SM Watchlist: {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to add to watchlist: {e}")
            
        except Exception as e:
            logger.error(f"Save sniper error: {e}")
        finally:
            session.close()

    def start(self):
        if self.is_running: return
        self.update_config()
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
                # 1. Визначення режиму
                if self.check_market_conditions():
                    
                    # 2. Пошук
                    candidates = self.find_whale_tracks()
                    
                    if candidates:
                        # 3. Полювання
                        self.hunt_targets(candidates)
                    else:
                        self.status = f"No Whale Volume found ({self.market_mode})"
                        
                    self.last_run_time = datetime.now().strftime("%H:%M:%S")
                    self.status = f"Waiting next cycle... ({self.market_mode})"
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


# ============================================================================
#                    COORDINATOR INTEGRATION
# ============================================================================

def register_with_coordinator():
    """Реєструє Whale SNIPER з координатором сканерів"""
    try:
        from scanner_coordinator import scanner_coordinator, ScannerType
        
        def scan_wrapper():
            """Обгортка для сканування - один цикл"""
            if sniper_bot.check_market_conditions():
                candidates = sniper_bot.find_whale_tracks()
                if candidates:
                    sniper_bot.hunt_targets(candidates)
        
        scanner_coordinator.set_scan_function(ScannerType.WHALE_SNIPER, scan_wrapper)
        logger.info("✅ Whale SNIPER registered with Coordinator")
        
    except ImportError:
        logger.warning("Scanner Coordinator not available")
    except Exception as e:
        logger.error(f"Coordinator registration error: {e}")


# Автореєстрація при імпорті
register_with_coordinator()
