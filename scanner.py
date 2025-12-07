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
        """
        Отримує свічки для розрахунку RSI - 100% сумісність з TradingView.
        
        ✅ КЛЮЧОВІ МОМЕНТИ:
        1. Порядок: Старі → Нові (хронологічний) - обов'язково для RSI
        2. Виключаємо останню незакриту свічку
        3. Беремо 200+ свічок для "прогріву" RSI (як TradingView)
        4. Правильна прив'язка до сітки часу для 45хв
        """
        try:
            # Мапинг TF
            tf_map = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            req_tf = tf_map.get(str(timeframe), '240')
            
            # ✅ Беремо 200 свічок для правильного "прогріву" RSI (як TradingView)
            req_limit = max(limit, 200)
            
            # Якщо потрібен 45хв, беремо в 3 рази більше 15хв свічок
            if str(timeframe) == '45':
                req_limit = req_limit * 3
            
            r = self.bot.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=req_limit)
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','volume','turnover'])
                
                # Конвертація типів
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['open'] = df['open'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['turnover'] = df['turnover'].astype(float)
                df['time'] = pd.to_numeric(df['time'])
                df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                
                # ✅ Сортуємо: Старі → Нові (ОБОВ'ЯЗКОВО для RSI!)
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # === ЛОГІКА РЕСЕМПЛІНГУ для 45хв ===
                if str(timeframe) == '45':
                    df.set_index('datetime', inplace=True)
                    
                    df = df.resample(
                        '45min',
                        origin='start_day',
                        label='left',
                        closed='left'
                    ).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'turnover': 'sum',
                        'time': 'first'
                    })
                    
                    df.dropna(inplace=True)
                    df = df.reset_index(drop=True)
                
                # ✅ Виключаємо останню свічку (незакрита) - TradingView так робить!
                if len(df) > 1:
                    df = df.iloc[:-1].reset_index(drop=True)
                
                # ✅ Повертаємо в хронологічному порядку (Старі → Нові)
                return df
        except Exception as e:
            logger.error(f"Fetch candles error {symbol} TF={timeframe}: {e}")
        return None

    def get_coin_data(self, symbol):
        """Отримує дані про монету з кешу"""
        return self.data.get(symbol, {})
    
    def get_current_rsi(self, symbol):
        """Отримує поточний RSI для монети"""
        return self.data.get(symbol, {}).get('rsi', 0)

    def monitor(self):
        """
        Монітор активних позицій з розрахунком RSI та ATR Trailing.
        Використовує ПРАВИЛЬНО ПРИВ'ЯЗАНІ свічки = ТОЧНИЙ RSI!
        Розрахунки на НАЛАШТОВУВАНОМУ робочому таймфреймі (LTF)
        """
        active_pos = self.get_active()
        active_syms = [p['symbol'] for p in active_pos]
        
        # Чистка кешу
        for k in list(self.data.keys()):
            if k not in active_syms: 
                del self.data[k]
        
        if not active_pos: return

        # === НАЛАШТОВАНІ ПАРАМЕТРИ ===
        exit_tf = settings.get("exit_ltf", "45")         # ✨ НОВЕ: LTF для розрахунків
        trailing_on = settings.get("trailing_enabled", False)
        trailing_trigger_rsi = float(settings.get("trailing_rsi_activation", 65))
        atr_len = int(settings.get("trailing_atr_length", 14))
        atr_mult = float(settings.get("trailing_atr_multiplier", 2.5))  # ✨ НАЛАШТОВУЄТЬСЯ
        trailing_delay = float(settings.get("trailing_activation_delay", 5))  # ✨ Затримка (хв)

        for p in active_pos:
            s = p['symbol']
            side = p['side'] # Buy / Sell
            current_price = float(p['avgPrice'])
            
            # Отримуємо поточну ціну (Last Price)
            last_price = 0.0
            try:
                last_price = float(self.bot.get_price(s))
            except: 
                last_price = current_price

            # Отримуємо поточний Stop Loss (якщо є)
            current_sl = float(p.get('stopLoss', 0.0))

            if s not in self.data: 
                self.data[s] = {
                    'rsi': 0, 
                    'exit_status': 'Safe', 
                    'exit_details': '-', 
                    'trailing_active': False,
                    'entry_sl': float(p.get('stopLoss', 0)),       # ✨ Початковий SL
                    'position_time': int(p.get('createdTime', 0))  # ✨ Час відкриття
                }

            # 1. Fetch Data з ПРАВИЛЬНОЮ ПРИВ'ЯЗКОЮ ✅
            df = self.fetch_candles(s, exit_tf, limit=atr_len + 50)
            
            if df is not None and len(df) >= 20:  # ✨ Знизили вимогу з 64 на 20 свічок
                # 2. Calc Indicators (ПРОФЕСІЙНІ МЕТОД Wilder's!)
                # ✅ ВАЖЛИВО: На ПРАВИЛЬНИХ свічках з КЛАСИЧНИМ Wilder's методом
                # ✅ Результат = ТОЧНО як у TradingView та Bybit!
                rsi_val = simple_rsi(df['close'], period=14)
                atr_val = simple_atr(df['high'], df['low'], df['close'], period=atr_len)
                
                self.data[s]['rsi'] = int(round(rsi_val))  # ✨ Округляємо до цілого числа

                status = "Safe"
                details = f"RSI: {round(rsi_val, 1)}"
                
                # --- ЛОГІКА ATR TRAILING ---
                if trailing_on:
                    is_active = self.data[s].get('trailing_active', False)
                    
                    # A. Перевірка Активації ("Пастка")
                    if not is_active:
                        # ✨ ПЕРЕВІРКА 1: Затримка після входу
                        pos_time = self.data[s].get('position_time', 0)
                        if pos_time > 0:
                            age_minutes = (time.time() * 1000 - pos_time) / 60000
                            if age_minutes < trailing_delay:
                                # Позиція ще молода - не активуємо trailing
                                status = "Safe"
                                details = f"RSI: {round(rsi_val, 1)} | Wait {int(trailing_delay - age_minutes)}m"
                                self.data[s]['exit_status'] = status
                                self.data[s]['exit_details'] = details
                                continue  # Пропускаємо цю позицію
                        
                        # RSI перевірка (після затримки)
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
                        entry_sl = self.data[s].get('entry_sl', 0)  # ✨ Початковий SL

                        if side == "Buy":
                            # Long: SL = Price - (ATR * Mult)
                            calc_sl = last_price - (atr_val * atr_mult)
                            # ✨ ПЕРЕВІРКА 2: Не гірше entry SL
                            if entry_sl > 0:
                                calc_sl = max(calc_sl, entry_sl)
                            # Рухаємо ТІЛЬКИ вгору
                            if calc_sl > current_sl:
                                new_sl = calc_sl
                                should_update = True

                        elif side == "Sell":
                            # Short: SL = Price + (ATR * Mult)
                            calc_sl = last_price + (atr_val * atr_mult)
                            # ✨ ПЕРЕВІРКА 2: Не гірше entry SL
                            if entry_sl > 0:
                                calc_sl = min(calc_sl, entry_sl)
                            # Рухаємо ТІЛЬКИ вниз (зменшуємо значення для шорта)
                            if current_sl == 0 or calc_sl < current_sl:
                                new_sl = calc_sl
                                should_update = True

                        if should_update and new_sl > 0:
                            success = self.bot.update_sl(s, new_sl)
                            if success:
                                logger.info(f"⛓️ Trailing SL updated for {s}: {current_sl} -> {new_sl}")
                                details += " | SL Upd ✅"

                # --- Візуалізація для UI ---
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

    def get_market_pressure(self, s): 
        """Заповнювач для сумісності"""
        return 0
        
    def get_active_symbols(self): 
        """Отримує активні символи"""
        return self.get_active()
