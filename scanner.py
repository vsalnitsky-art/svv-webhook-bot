#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
from settings_manager import settings
from bot import bot_instance
from indicators import simple_rsi, simple_atr

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.data = {}

    def start(self):
        """Запускає фоновий потік моніторингу. Цей метод викликається з main_app.py"""
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("✅ Enhanced Market Scanner & Trailing Started")

    def loop(self):
        while True:
            try:
                self.monitor()
            except Exception as e:
                # logger.error(f"Scanner loop error: {e}")
                pass
            time.sleep(5) # Перевіряємо кожні 5 сек для швидкого трейлінгу

    def get_active(self):
        try:
            r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                return [p for p in r['result']['list'] if float(p['size']) > 0]
        except: pass
        return []

    def fetch_candles(self, symbol, timeframe, limit=50):
        try:
            # Мапинг TF
            tf_map = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            req_tf = tf_map.get(str(timeframe), '240')
            
            # === ВИПРАВЛЕННЯ 1: Збільшуємо ліміт для склейки 45м ===
            req_limit = limit * 3 if str(timeframe) == '45' else limit
            
            r = self.bot.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=req_limit)
            if r['retCode'] == 0:
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','vol','to'])
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['open'] = df['open'].astype(float) 
                
                # Конвертуємо час для правильного ресемплінгу
                df['time'] = pd.to_numeric(df['time'])
                df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                
                # Перевертаємо: старі -> нові
                df = df.sort_values('datetime').reset_index(drop=True)

                # === ВИПРАВЛЕННЯ 2: Логіка склейки (Resample) для 45хв ===
                if str(timeframe) == '45':
                    df.set_index('datetime', inplace=True)
                    # origin='start_day' гарантує, що сітка починається з 00:00 (як на TradingView)
                    df_45 = df.resample('45min', origin='start_day', closed='left', label='left').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last'
                    })
                    df_45.dropna(inplace=True)
                    df = df_45.reset_index(drop=True)

                return df
        except Exception as e: 
            pass
        return None

    def monitor(self):
        active_pos = self.get_active()
        active_syms = [p['symbol'] for p in active_pos]
        
        # Чистка кешу
        for k in list(self.data.keys()):
            if k not in active_syms: 
                del self.data[k]
        
        if not active_pos: return

        # Глобальні параметри
        tf = settings.get("htfSelection", "240")
        trailing_on = settings.get("trailing_enabled", False)
        trailing_trigger_rsi = float(settings.get("trailing_rsi_activation", 65))
        atr_len = int(settings.get("trailing_atr_length", 14))
        atr_mult = float(settings.get("trailing_atr_multiplier", 2.5))

        for p in active_pos:
            s = p['symbol']
            side = p['side'] # Buy / Sell
            current_price = float(p['avgPrice']) # Або lastPrice, але для SL краще дивитись на current market price
            
            # Отримуємо поточну ціну (Last Price)
            last_price = 0.0
            try:
                last_price = float(self.bot.get_price(s))
            except: 
                last_price = current_price

            # Отримуємо поточний Stop Loss (якщо є)
            current_sl = float(p.get('stopLoss', 0.0))

            if s not in self.data: 
                self.data[s] = {'rsi': 0, 'exit_status': 'Safe', 'exit_details': '-', 'trailing_active': False}

            # 1. Fetch Data
            # Беремо більше свічок, щоб після склейки залишилось достатньо для ATR/RSI
            df = self.fetch_candles(s, tf, limit=(atr_len + 50))
            
            if df is not None and len(df) > atr_len:
                # 2. Calc Indicators (без pandas_ta - fallback)
                rsi_val = simple_rsi(df['close'], period=14)
                atr_val = simple_atr(df['high'], df['low'], df['close'], period=atr_len)
                
                self.data[s]['rsi'] = round(rsi_val, 1)

                status = "Safe"
                details = f"RSI: {round(rsi_val, 1)}"
                
                # --- ЛОГІКА ATR TRAILING ---
                if trailing_on:
                    is_active = self.data[s].get('trailing_active', False)
                    
                    # A. Перевірка Активації ("Пастка")
                    if not is_active:
                        if side == "Buy" and rsi_val >= trailing_trigger_rsi:
                            self.data[s]['trailing_active'] = True
                            is_active = True
                            logger.info(f"🪤 ATR Trailing ACTIVATED for {s} (Long). RSI: {rsi_val}")
                        elif side == "Sell" and rsi_val <= (100 - trailing_trigger_rsi):
                            self.data[s]['trailing_active'] = True
                            is_active = True
                            logger.info(f"🪤 ATR Trailing ACTIVATED for {s} (Short). RSI: {rsi_val}")

                    # B. Розрахунок та Оновлення Стопу
                    if is_active:
                        status = "TRAILING 🚀"
                        details = f"ATR: {round(atr_val, 4)}"
                        
                        new_sl = 0.0
                        should_update = False

                        if side == "Buy":
                            # Long: SL = Price - (ATR * Mult)
                            calc_sl = last_price - (atr_val * atr_mult)
                            # Рухаємо ТІЛЬКИ вгору
                            if calc_sl > current_sl:
                                new_sl = calc_sl
                                should_update = True
                        
                        elif side == "Sell":
                            # Short: SL = Price + (ATR * Mult)
                            calc_sl = last_price + (atr_val * atr_mult)
                            # Рухаємо ТІЛЬКИ вниз 
                            if current_sl == 0 or calc_sl < current_sl:
                                new_sl = calc_sl
                                should_update = True

                        if should_update and new_sl > 0:
                            success = self.bot.update_sl(s, new_sl)
                            if success:
                                logger.info(f"⛓️ Trailing SL updated for {s}: {current_sl} -> {new_sl}")
                                details += " | SL Upd ✅"

                # --- Візуалізація для UI (Старий код) ---
                lb = float(settings.get('obt_entryRsiOversold', 30))
                ub = float(settings.get('obt_entryRsiOverbought', 70))
                
                if not self.data[s].get('trailing_active'):
                    if side == "Buy" and rsi_val >= ub: 
                        status = "Warning"
                        details += " (High)"
                    elif side == "Sell" and rsi_val <= lb: 
                        status = "Warning"
                        details += " (Low)"
                
                self.data[s]['exit_status'] = status
                self.data[s]['exit_details'] = details
            
            # Невеликий сліп між монетами
            time.sleep(0.2)

    def get_coin_data(self, s):
        # Повертає словник, навіть якщо монети немає
        return self.data.get(s, {'rsi': 0, 'exit_status': 'Safe', 'exit_details': '-'})
        
    def get_current_rsi(self, s): return self.data.get(s, {}).get('rsi', 0)
    def get_market_pressure(self, s): return 0
    def get_active_symbols(self): return self.get_active()