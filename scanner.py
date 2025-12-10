#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SVV Webhook Bot - Enhanced Market Scanner
=============================================
Версія: 3.2 (Smart TP Monitor)

Функціонал:
1. Моніторинг активних позицій
2. Smart TP Tracking (50/25/25)
3. Auto Break-Even після TP1
4. Auto Trailing після TP2
5. RSI розрахунок для виходу
"""
import threading
import time
import logging
import pandas as pd
from settings_manager import settings
from bot import bot_instance
from indicators import simple_rsi, simple_atr

logger = logging.getLogger(__name__)


class EnhancedMarketScanner:
    """
    🎯 Smart Position Monitor
    
    Відстежує позиції та автоматично:
    - Переміщує SL в Break-Even після TP1 (50%)
    - Активує Trailing Stop після TP2 (25%)
    - Залишок 25% працює на Trailing
    """
    
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.data = {}  # {symbol: position_data}

    def start(self):
        """Запускає фоновий потік моніторингу"""
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("✅ Enhanced Market Scanner Started (Smart TP Mode)")

    def loop(self):
        """Головний цикл моніторингу"""
        while True:
            try:
                self.monitor()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            time.sleep(5)  # Перевіряємо кожні 5 сек

    def get_active(self):
        """Отримує список активних позицій"""
        try:
            r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                return [p for p in r['result']['list'] if float(p['size']) > 0]
        except Exception as e:
            logger.error(f"Get active positions error: {e}")
        return []

    def fetch_candles(self, symbol, timeframe, limit=200):
        """
        Завантаження свічок для RSI/ATR.
        
        Важливо:
        - Порядок: Старі → Нові
        - Виключаємо останню незакриту свічку
        - Мінімум 200 свічок для точного RSI
        """
        try:
            tf_map = {'5':'5', '15':'15', '30':'30', '45':'15', 
                      '60':'60', '120':'120', '240':'240', 'D':'D', 'W':'W'}
            req_tf = tf_map.get(str(timeframe), '60')
            
            req_limit = max(limit, 200)
            if str(timeframe) == '45':
                req_limit = req_limit * 3
            if req_limit > 1000:
                req_limit = 1000
            
            r = self.bot.session.get_kline(
                category="linear", 
                symbol=symbol, 
                interval=req_tf, 
                limit=req_limit
            )
            
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(
                    r['result']['list'], 
                    columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                )
                
                cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                df[cols] = df[cols].astype(float)
                df['time'] = pd.to_numeric(df['time'])
                df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                
                # Сортуємо: Старі → Нові
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # Ресемплінг для 45хв
                if str(timeframe) == '45':
                    df.set_index('datetime', inplace=True)
                    df = df.resample('45min', origin='start_day', label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 
                        'close': 'last', 'volume': 'sum', 'turnover': 'sum', 'time': 'first'
                    })
                    df.dropna(inplace=True)
                    df = df.reset_index(drop=True)
                
                # Виключаємо останню незакриту свічку
                if len(df) > 1:
                    df = df.iloc[:-1].reset_index(drop=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Fetch candles error {symbol}: {e}")
        return None

    def get_coin_data(self, symbol):
        """Отримує дані про монету з кешу"""
        return self.data.get(symbol, {})
    
    def get_current_rsi(self, symbol):
        """Отримує поточний RSI для монети"""
        return self.data.get(symbol, {}).get('rsi', 0)

    def monitor(self):
        """
        🎯 ГОЛОВНА ЛОГІКА МОНІТОРИНГУ
        
        Smart TP Algorithm:
        1. При відкритті - запам'ятовуємо initial_qty
        2. Коли qty ≤ 50% initial → TP1 виконався → SL в BE
        3. Коли qty ≤ 25% initial → TP2 виконався → Trailing ON
        """
        active_pos = self.get_active()
        active_syms = [p['symbol'] for p in active_pos]
        
        # Очищаємо дані для закритих позицій
        for k in list(self.data.keys()):
            if k not in active_syms: 
                del self.data[k]
        
        if not active_pos:
            return

        # === НАЛАШТУВАННЯ ===
        exit_tf = settings.get("exit_ltf", "60")
        tp_mode = settings.get("tp_mode", "Smart_TP")
        
        # Trailing параметри (використовуються після TP2)
        atr_len = int(settings.get("trailing_atr_length", 14))
        atr_mult = float(settings.get("trailing_atr_multiplier", 2.5))

        for p in active_pos:
            s = p['symbol']
            side = p['side']
            entry_price = float(p['avgPrice'])
            current_qty = float(p['size'])
            position_idx = int(p.get('positionIdx', 0))
            current_sl = float(p.get('stopLoss', 0.0))
            
            # Отримуємо поточну ціну
            try:
                last_price = float(self.bot.get_price(s))
            except:
                last_price = entry_price

            # === ІНІЦІАЛІЗАЦІЯ ДАНИХ ПОЗИЦІЇ ===
            if s not in self.data:
                self.data[s] = {
                    'rsi': 0,
                    'exit_status': 'Active',
                    'exit_details': 'Monitoring...',
                    
                    # 🎯 Smart TP Tracking
                    'initial_qty': current_qty,      # Початковий розмір позиції
                    'entry_price': entry_price,
                    'entry_sl': current_sl,
                    'position_time': int(p.get('createdTime', 0)),
                    
                    # TP/BE/Trailing стани
                    'tp1_hit': False,      # TP1 виконався (50% закрито)
                    'tp2_hit': False,      # TP2 виконався (ще 25% закрито)
                    'be_set': False,       # Break-Even встановлено
                    'trailing_active': False,  # Trailing активний
                    'last_sl_update': 0    # Останнє значення SL
                }
                logger.info(f"📊 New position tracked: {s} {side} qty={current_qty} entry={entry_price}")

            pos_data = self.data[s]
            initial_qty = pos_data['initial_qty']
            
            # === SMART TP TRACKING (для режимів Smart_TP / Fixed_1_50) ===
            if tp_mode in ["Smart_TP", "Fixed_1_50"]:
                
                # Визначаємо скільки % позиції залишилось
                qty_ratio = current_qty / initial_qty if initial_qty > 0 else 1.0
                
                # --- TP1 CHECK (qty ≤ 55% означає що TP1 виконався) ---
                if not pos_data['tp1_hit'] and qty_ratio <= 0.55:
                    pos_data['tp1_hit'] = True
                    logger.info(f"✅ TP1 HIT: {s} - 50% closed. Remaining: {round(qty_ratio*100)}%")
                    
                    # → Переміщуємо SL в Break-Even
                    if not pos_data['be_set']:
                        be_buffer = 0.001  # 0.1% буфер
                        
                        if side == "Buy":
                            be_price = entry_price * (1 + be_buffer)
                            # BE має бути вище поточного SL
                            if current_sl == 0 or be_price > current_sl:
                                if self.bot.update_sl(s, be_price, position_idx):
                                    pos_data['be_set'] = True
                                    pos_data['last_sl_update'] = be_price
                                    logger.info(f"🛡️ BE SET: {s} SL moved to {be_price} (+0.1%)")
                        else:  # Sell
                            be_price = entry_price * (1 - be_buffer)
                            # BE має бути нижче поточного SL
                            if current_sl == 0 or be_price < current_sl:
                                if self.bot.update_sl(s, be_price, position_idx):
                                    pos_data['be_set'] = True
                                    pos_data['last_sl_update'] = be_price
                                    logger.info(f"🛡️ BE SET: {s} SL moved to {be_price} (-0.1%)")
                
                # --- TP2 CHECK (qty ≤ 30% означає що TP2 виконався) ---
                if pos_data['tp1_hit'] and not pos_data['tp2_hit'] and qty_ratio <= 0.30:
                    pos_data['tp2_hit'] = True
                    pos_data['trailing_active'] = True
                    logger.info(f"✅ TP2 HIT: {s} - 75% total closed. Trailing ACTIVATED for remaining {round(qty_ratio*100)}%")

            # === FETCH DATA FOR INDICATORS ===
            df = self.fetch_candles(s, exit_tf, limit=atr_len + 50)
            
            if df is not None and len(df) >= 15:
                # Розрахунок індикаторів
                rsi_val = simple_rsi(df['close'], period=14)
                atr_val = simple_atr(df['high'], df['low'], df['close'], period=atr_len)
                
                pos_data['rsi'] = int(round(rsi_val))
                
                # === ВИЗНАЧЕННЯ СТАТУСУ ===
                status = "Active"
                details = f"RSI: {round(rsi_val, 1)}"
                
                if pos_data['tp1_hit'] and not pos_data['tp2_hit']:
                    status = "TP1 ✓ BE"
                    details = f"RSI: {round(rsi_val, 1)} | Waiting TP2"
                
                # === TRAILING STOP LOGIC (тільки після TP2) ===
                if pos_data['trailing_active']:
                    status = "TRAILING 🚀"
                    details = f"ATR: {round(atr_val, 4)}"
                    
                    last_sl = pos_data.get('last_sl_update', current_sl)
                    new_sl = 0.0
                    should_update = False

                    if side == "Buy":
                        # Trailing SL = Price - (ATR × mult)
                        calc_sl = last_price - (atr_val * atr_mult)
                        
                        # SL не може бути нижче BE
                        be_level = entry_price * 1.001
                        calc_sl = max(calc_sl, be_level)
                        
                        # Оновлюємо тільки якщо новий SL ВИЩИЙ
                        if calc_sl > last_sl and (calc_sl - last_sl) / last_sl > 0.001:
                            new_sl = calc_sl
                            should_update = True

                    elif side == "Sell":
                        # Trailing SL = Price + (ATR × mult)
                        calc_sl = last_price + (atr_val * atr_mult)
                        
                        # SL не може бути вище BE
                        be_level = entry_price * 0.999
                        calc_sl = min(calc_sl, be_level)
                        
                        # Оновлюємо тільки якщо новий SL НИЖЧИЙ
                        if last_sl == 0 or (calc_sl < last_sl and (last_sl - calc_sl) / last_sl > 0.001):
                            new_sl = calc_sl
                            should_update = True

                    # Оновлюємо SL якщо потрібно
                    if should_update and new_sl > 0:
                        success = self.bot.update_sl(s, new_sl, position_idx)
                        if success:
                            pos_data['last_sl_update'] = new_sl
                            logger.info(f"⛓️ TRAILING: {s} SL updated {last_sl:.4f} → {new_sl:.4f}")
                            details += " | SL ✓"
                        else:
                            details += " | SL OK"
                    else:
                        details += " | SL OK"
                
                # === RSI WARNING (для активних позицій без trailing) ===
                if not pos_data['trailing_active']:
                    lb = float(settings.get('obt_entryRsiOversold', 30))
                    ub = float(settings.get('obt_entryRsiOverbought', 70))
                    
                    if side == "Buy" and rsi_val >= ub:
                        status = "⚠️ RSI High"
                        details += f" (≥{ub})"
                    elif side == "Sell" and rsi_val <= lb:
                        status = "⚠️ RSI Low"
                        details += f" (≤{lb})"
                
                pos_data['exit_status'] = status
                pos_data['exit_details'] = details
            
            else:
                pos_data['exit_status'] = 'No Data'
                pos_data['exit_details'] = 'Waiting...'
            
            # Затримка між монетами
            time.sleep(0.2)

    def get_market_pressure(self, s):
        """Заповнювач для сумісності"""
        return 0
        
    def get_active_symbols(self):
        """Отримує активні символи"""
        return self.get_active()
