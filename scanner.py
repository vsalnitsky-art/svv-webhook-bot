#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
import pandas_ta as ta
from settings_manager import settings
from bot import bot_instance
# Імпорт стратегії для перевірки виходу
from strategy_ob_trend import ob_trend_strategy

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.running = True
        
        # Кеш даних для активних монет (включаючи статус виходу)
        self.active_coins_data = {} 
        self.scan_interval = 10 # Трохи рідше, бо вантажимо HTF

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info(f"🚀 Active Position Monitor & Smart Exit Started")
    
    def loop(self):
        while self.running:
            try:
                self.monitor_positions()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            time.sleep(self.scan_interval)
    
    def get_active_symbols(self):
        """✅ ВИПРАВЛЕНО: Повертає ПОЗИЦІЇ (об'єкти), а не тільки символи"""
        positions = []
        try:
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        positions.append(p)
        except Exception as e:
            logger.error(f"Active symbols error: {e}")
        return positions
    
    def get_active_symbols_list(self):
        """✅ ВИПРАВЛЕНО: Повертає ТІЛЬКИ СИМВОЛИ як список"""
        return [p['symbol'] for p in self.get_active_symbols()]

    def fetch_htf_candles(self, symbol):
        """Завантажує свічки Глобального ТФ для аналізу виходу"""
        try:
            htf = settings.get("htfSelection")
            # Мапінг для Bybit
            tf_map = {'60': '60', '240': '240', 'D': 'D'}
            req_tf = tf_map.get(str(htf), '240')
            
            resp = self.bot.session.get_kline(
                category="linear", symbol=symbol, interval=req_tf, limit=50 
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                data = resp['result']['list']
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None

    def monitor_positions(self):
        active_positions = self.get_active_symbols()
        target_symbols = self.get_active_symbols_list()  # ✅ ВИПРАВЛЕНО: Використовуємо новий метод
        
        # Чистка кешу
        current_keys = list(self.active_coins_data.keys())
        for k in current_keys:
            if k not in target_symbols:
                del self.active_coins_data[k]

        if not active_positions: return

        # Чи увімкнена стратегія виходу?
        smart_exit_enabled = settings.get("exit_enableStrategy", False)

        for pos in active_positions:
            symbol = pos['symbol']
            side = pos['side'] # Buy / Sell
            
            if symbol not in self.active_coins_data:
                self.active_coins_data[symbol] = {
                    'rsi': 0, 'pressure': 0, 'exit_status': 'Safe', 'exit_details': '-'
                }
            
            # 1. Завантаження даних HTF (якщо стратегія активна або для відображення RSI)
            df_htf = self.fetch_htf_candles(symbol)
            
            # --- РОЗУМНИЙ ВИХІД ---
            exit_info = {'close': False, 'reason': '', 'details': {}}
            
            if df_htf is not None:
                # Викликаємо стратегію
                exit_info = ob_trend_strategy.check_exit_signal(df_htf, side)
                
                # Оновлюємо дані для UI
                self.active_coins_data[symbol]['rsi'] = exit_info['details'].get('rsi', 0)
                
                # Формуємо статус
                if exit_info['close']:
                    self.active_coins_data[symbol]['exit_status'] = 'EXIT NOW'
                    self.active_coins_data[symbol]['exit_details'] = exit_info['reason']
                    
                    # === ВИКОНАННЯ ЗАКРИТТЯ ===
                    if smart_exit_enabled:
                        logger.info(f"🚨 SMART EXIT TRIGGERED: {symbol} ({side}) -> {exit_info['reason']}")
                        self.bot.place_order({
                            "action": "Close",
                            "symbol": symbol,
                            "direction": "Long" if side == "Buy" else "Short"
                        })
                else:
                    # Статуси попередження
                    rsi_val = exit_info['details'].get('rsi', 50)
                    limit_long = float(settings.get('exit_rsiOverbought', 70))
                    limit_short = float(settings.get('exit_rsiOversold', 30))
                    
                    status = "Safe"
                    if side == "Buy" and rsi_val >= (limit_long - 5): status = "Warning"
                    if side == "Sell" and rsi_val <= (limit_short + 5): status = "Warning"
                    
                    self.active_coins_data[symbol]['exit_status'] = status
                    self.active_coins_data[symbol]['exit_details'] = f"RSI: {rsi_val}"

            time.sleep(0.5)

    # === GETTERS FOR UI ===
    def get_coin_data(self, symbol):
        return self.active_coins_data.get(symbol, {})

    def get_current_rsi(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('rsi', 0)

    def get_market_pressure(self, symbol):
        # Pressure поки спрощено (можна відновити стару логіку, якщо треба),
        # але тут фокус на Exit Strategy. Повернемо 0 або старе значення.
        return self.active_coins_data.get(symbol, {}).get('pressure', 0)
    
    def get_exit_status(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('exit_status', 'N/A')
    
    def get_exit_details(self, symbol):
        return self.active_coins_data.get(symbol, {}).get('exit_details', '-')