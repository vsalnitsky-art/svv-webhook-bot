#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SVV Webhook Bot - OB Trend Strategy
=======================================
Версія: 3.0 (RSI 100% TradingView Compatible)

Стратегія Order Blocks + Trend з професійним RSI (Wilder's Smoothing).
Всі індикатори розраховуються без TA-Lib.
"""
import pandas as pd
import numpy as np
from settings_manager import settings
from indicators import (
    calculate_rsi_series, 
    calculate_hma, 
    calculate_obv, 
    calculate_sma, 
    calculate_ema,
    calculate_all_indicators
)


class OBTrendStrategy:
    """
    Order Block + Trend Strategy з RSI фільтрацією.
    
    Особливості:
    - RSI 100% ідентичний TradingView (Wilder's Smoothing)
    - HMA Cloud для визначення тренду
    - OBV для підтвердження об'єму
    - Order Blocks для точок входу
    """
    
    def __init__(self):
        pass

    def _get_param(self, key, default=None):
        """Отримання параметру з налаштувань"""
        val = settings.get(key)
        return val if val is not None else default

    def calculate_indicators(self, df):
        """
        🎯 Розрахунок ВСІХ індикаторів для стратегії.
        
        Використовує indicators.py для:
        - RSI (Wilder's Smoothing)
        - HMA (Hull Moving Average)
        - OBV (On-Balance Volume)
        """
        if df is None or len(df) < 50: 
            return df
            
        try:
            # === RSI (100% TradingView) ===
            rsi_len = int(self._get_param('obt_rsiLength', 14))
            df['rsi'] = calculate_rsi_series(df['close'], period=rsi_len)
            
            # === HMA Cloud ===
            fast_len = int(self._get_param('obt_cloudFastLen', 10))
            slow_len = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = calculate_hma(df['close'], period=fast_len)
            df['hma_slow'] = calculate_hma(df['close'], period=slow_len)
            
            # === OBV ===
            if 'volume' in df.columns:
                df['obv'] = calculate_obv(df['close'], df['volume'])
                
                # OBV MA для фільтру
                obv_entry_len = int(self._get_param('obt_obvEntryLen', 20))
                obv_exit_len = int(self._get_param('exit_obvLength', 10))
                
                df['obv_ma'] = calculate_sma(df['obv'], period=obv_entry_len)
                df['obv_exit_ma'] = calculate_ema(df['obv'], period=obv_exit_len)
            
        except Exception as e:
            pass
            
        return df

    def find_order_blocks(self, df):
        """
        Пошук Order Blocks (Swing High/Low).
        
        Order Block - це зона, де великий гравець виставляв ордери.
        """
        obs = {'buy': [], 'sell': []}
        
        if df is None or len(df) < 100: 
            return obs
            
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        
        for i in range(swing, len(subset) - swing):
            cl, ch = subset['low'].iloc[i], subset['high'].iloc[i]
            
            # === BUY ORDER BLOCK (Swing Low) ===
            is_swing_low = all(
                subset['low'].iloc[i-j] >= cl and subset['low'].iloc[i+j] >= cl 
                for j in range(1, swing+1)
            )
            
            if is_swing_low:
                # Перевірка пробою вгору (bullish confirmation)
                if subset['close'].iloc[i+1] > ch: 
                    obs['buy'].append({
                        'top': ch, 
                        'bottom': cl, 
                        'created_at': subset['time'].iloc[i]
                    })
            
            # === SELL ORDER BLOCK (Swing High) ===
            is_swing_high = all(
                subset['high'].iloc[i-j] <= ch and subset['high'].iloc[i+j] <= ch 
                for j in range(1, swing+1)
            )
            
            if is_swing_high:
                # Перевірка пробою вниз (bearish confirmation)
                if subset['close'].iloc[i+1] < cl: 
                    obs['sell'].append({
                        'top': ch, 
                        'bottom': cl, 
                        'created_at': subset['time'].iloc[i]
                    })
        
        # Повертаємо останні 3 OB кожного типу
        return {
            'buy': obs['buy'][-3:], 
            'sell': obs['sell'][-3:]
        }

    def check_exit_signal(self, df_htf, position_side):
        """
        Перевірка сигналу на вихід з позиції.
        
        Умови виходу:
        - RSI досяг екстремуму (Overbought/Oversold)
        - OBV дивергенція
        - Confluence (RSI + OBV)
        """
        res = {'close': False, 'reason': '', 'details': {}}
        
        if df_htf is None: 
            return res
            
        # Розраховуємо індикатори
        df = self.calculate_indicators(df_htf)
        last = df.iloc[-1]
        
        # Отримуємо значення індикаторів
        rsi = last.get('rsi', 50)
        obv = last.get('obv', 0)
        obv_ma = last.get('obv_exit_ma', 0)
        
        res['details'] = {
            'rsi': round(rsi, 1), 
            'obv_cross': 'UP' if obv > obv_ma else 'DOWN'
        }
        
        # Рівні RSI для виходу
        limit_buy = float(self._get_param('exit_rsiOverbought', 70))
        limit_sell = float(self._get_param('exit_rsiOversold', 30))

        if position_side == 'Buy':
            # Long позиція - шукаємо сигнал на закриття
            if rsi >= limit_buy: 
                res.update({'close': True, 'reason': 'RSI Max'})
            elif rsi >= limit_buy and obv < obv_ma: 
                res.update({'close': True, 'reason': 'Confluence'})
                
        elif position_side == 'Sell':
            # Short позиція - шукаємо сигнал на закриття
            if rsi <= limit_sell: 
                res.update({'close': True, 'reason': 'RSI Min'})
            elif rsi <= limit_sell and obv > obv_ma: 
                res.update({'close': True, 'reason': 'Confluence'})
                
        return res

    def analyze(self, df_ltf, df_htf):
        """
        🎯 Головний аналіз для генерації торгових сигналів.
        
        Логіка:
        1. HTF (Higher Time Frame) - визначає дозволений напрямок (тренд)
        2. LTF (Lower Time Frame) - генерує точні входи
        
        Фільтри:
        - Cloud Filter (HMA)
        - OBV Filter
        - RSI Filter
        """
        signals = []
        
        # Розраховуємо індикатори для обох ТФ
        df_h = self.calculate_indicators(df_htf)
        df_l = self.calculate_indicators(df_ltf)
        
        if df_h is None or df_l is None: 
            return []
        
        row_h = df_h.iloc[-1]  # Останній рядок HTF
        row_l = df_l.iloc[-1]  # Останній рядок LTF
        curr_price = row_l['close']
        
        # === 1. ВИЗНАЧЕННЯ ДОЗВОЛЕНИХ НАПРЯМКІВ (TREND FILTER) ===
        allow_long = True
        allow_short = True
        
        # Читаємо налаштування фільтрів
        use_cloud = self._get_param('obt_useCloudFilter', True)
        use_obv = self._get_param('obt_useObvFilter', True)
        use_rsi = self._get_param('obt_useRsiFilter', True)
        
        is_trend_filter_active = use_cloud or use_obv or use_rsi

        if is_trend_filter_active:
            # === CLOUD FILTER (HMA) ===
            if use_cloud:
                hma_fast = row_h.get('hma_fast', 0)
                hma_slow = row_h.get('hma_slow', 0)
                
                # Якщо HMA Fast < HMA Slow - даунтренд, блокуємо лонги
                if hma_fast <= hma_slow: 
                    allow_long = False
                # Якщо HMA Fast > HMA Slow - аптренд, блокуємо шорти
                if hma_fast >= hma_slow: 
                    allow_short = False
            
            # === OBV FILTER ===
            if use_obv:
                obv = row_h.get('obv', 0)
                obv_ma = row_h.get('obv_ma', 0)
                
                # OBV нижче MA - ведмежий сигнал
                if obv <= obv_ma: 
                    allow_long = False
                # OBV вище MA - бичачий сигнал
                if obv >= obv_ma: 
                    allow_short = False

            # === HTF RSI FILTER (Загальний тренд) ===
            if use_rsi:
                htf_rsi = row_h.get('rsi', 50)
                
                # RSI > 55 - перекуплений, не входимо в лонг
                if htf_rsi > 55: 
                    allow_long = False
                # RSI < 45 - перепроданий, не входимо в шорт
                if htf_rsi < 45: 
                    allow_short = False

        # === 2. ЛОГІКА ВХОДУ (TRIGGER) ===
        use_retest = self._get_param('obt_useOBRetest', False)
        rsi_val = row_l.get('rsi', 50)
        
        # --- LONG SIGNALS ---
        if allow_long:
            # RSI Entry Check: Якщо RSI Filter увімкнено - перевіряємо oversold
            rsi_oversold = float(self._get_param('obt_entryRsiOversold', 45))
            rsi_condition = (rsi_val <= rsi_oversold) if use_rsi else True
            
            if rsi_condition:
                signal_found = False
                sl = 0.0
                reason = ""
                
                if use_retest:
                    # Режим Retest - ціна повертається в зону OB
                    obs = self.find_order_blocks(df_l)
                    for ob in obs['buy']:
                        # Ціна в зоні Order Block (+0.3% буфер)
                        if ob['bottom'] <= curr_price <= (ob['top'] * 1.003):
                            signal_found = True
                            sl = ob['bottom'] * 0.995  # SL під OB
                            reason = "OB Retest"
                            break
                else:
                    # RAW MODE - просто перевіряємо наявність OB
                    obs = self.find_order_blocks(df_l)
                    if obs['buy']: 
                        signal_found = True
                        sl = curr_price * 0.985  # SL -1.5%
                        reason = "Trend+OB Exist"
                
                if signal_found:
                    signals.append({
                        'action': 'Buy', 
                        'price': curr_price, 
                        'rsi': round(rsi_val, 1), 
                        'reason': reason, 
                        'sl_price': sl
                    })

        # --- SHORT SIGNALS ---
        if allow_short:
            # RSI Entry Check: Якщо RSI Filter увімкнено - перевіряємо overbought
            rsi_overbought = float(self._get_param('obt_entryRsiOverbought', 55))
            rsi_condition = (rsi_val >= rsi_overbought) if use_rsi else True
            
            if rsi_condition:
                signal_found = False
                sl = 0.0
                reason = ""
                
                if use_retest:
                    # Режим Retest
                    obs = self.find_order_blocks(df_l)
                    for ob in obs['sell']:
                        # Ціна в зоні Order Block (-0.3% буфер)
                        if (ob['bottom'] * 0.997) <= curr_price <= ob['top']:
                            signal_found = True
                            sl = ob['top'] * 1.005  # SL над OB
                            reason = "OB Retest"
                            break
                else:
                    # RAW MODE
                    obs = self.find_order_blocks(df_l)
                    if obs['sell']:
                        signal_found = True
                        sl = curr_price * 1.015  # SL +1.5%
                        reason = "Trend+OB Exist"
                
                if signal_found:
                    signals.append({
                        'action': 'Sell', 
                        'price': curr_price, 
                        'rsi': round(rsi_val, 1), 
                        'reason': reason, 
                        'sl_price': sl
                    })

        return signals


# Singleton instance
ob_trend_strategy = OBTrendStrategy()
