#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pandas_ta as ta
import logging
from settings_manager import settings

logger = logging.getLogger(__name__)

class ObTrendStrategy:
    def __init__(self):
        pass

    def get_param(self, key, default=None):
        val = settings.get(key)
        return val if val is not None else default

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            # HMA Cloud
            fast_len = int(self.get_param('obt_cloudFastLen', 10))
            slow_len = int(self.get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            
            # RSI
            rsi_len = int(self.get_param('obt_rsiLength', 14))
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            obv_len = int(self.get_param('obt_obvEntryLen', 20))
            if 'obv' in df: 
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
            
            # ATR (для стопів)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
        except Exception as e:
            logger.error(f"Ind Error: {e}")
        return df

    def find_order_blocks(self, df):
        """Пошук OB для Smart Money та Retest стратегії"""
        if df is None or len(df) < 50: return {'buy': [], 'sell': []}
        
        bull_obs = []
        bear_obs = []
        swing = int(self.get_param('obt_swingLength', 5))
        
        # Спрощений алгоритм пошуку фракталів/свінгів
        subset = df.tail(300).reset_index(drop=True)
        if len(subset) < swing * 2: return {'buy': [], 'sell': []}

        for i in range(swing, len(subset) - swing):
            # Swing High
            is_high = True
            for j in range(1, swing + 1):
                if subset['high'][i] <= subset['high'][i-j] or subset['high'][i] <= subset['high'][i+j]:
                    is_high = False
                    break
            
            # Swing Low
            is_low = True
            for j in range(1, swing + 1):
                if subset['low'][i] >= subset['low'][i-j] or subset['low'][i] >= subset['low'][i+j]:
                    is_low = False
                    break
            
            if is_high:
                # Bearish OB (остання бича свічка перед падінням)
                # Шукаємо свічку з найвищим High в районі свінгу
                top = subset['high'][i]
                # Перевірка на пробій структури вниз (BOS) - спрощено: ціна пішла нижче low свінгу
                valid_ob = False
                for k in range(i+1, len(subset)):
                    if subset['close'][k] > top: # Invalidated
                        break
                    if subset['close'][k] < subset['low'][i] - (subset['high'][i] - subset['low'][i]): # Momentum down
                        valid_ob = True
                
                if valid_ob:
                    ob_candle = subset.iloc[i]
                    bear_obs.append({
                        'top': ob_candle['high'],
                        'bottom': ob_candle['low'],
                        'created_at': ob_candle['time']
                    })

            if is_low:
                # Bullish OB
                btm = subset['low'][i]
                valid_ob = False
                for k in range(i+1, len(subset)):
                    if subset['close'][k] < btm: # Invalidated
                        break
                    if subset['close'][k] > subset['high'][i] + (subset['high'][i] - subset['low'][i]): # Momentum up
                        valid_ob = True
                
                if valid_ob:
                    ob_candle = subset.iloc[i]
                    bull_obs.append({
                        'top': ob_candle['high'],
                        'bottom': ob_candle['low'],
                        'created_at': ob_candle['time']
                    })
        
        # Фільтруємо ті, що ще не пробиті
        curr_close = subset['close'].iloc[-1]
        active_bull = [b for b in bull_obs if curr_close > b['bottom']]
        active_bear = [b for b in bear_obs if curr_close < b['top']]
        
        return {'buy': active_bull[-5:], 'sell': active_bear[-5:]}

    def check_htf_filters(self, htf_row):
        """Перевірка тренду на старшому ТФ"""
        if htf_row is None or 'hma_fast' not in htf_row.index: 
            return {'bull': False, 'bear': False, 'details': {}}
            
        use_cloud = self.get_param('obt_useCloudFilter', True)
        use_rsi = self.get_param('obt_useRsiFilter', True)
        use_obv = self.get_param('obt_useObvFilter', True)

        # 1. Cloud
        cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
        cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
        
        # 2. RSI
        rsi_val = htf_row.get('rsi', 50)
        rsi_bull = (rsi_val <= 60) if use_rsi else True # Не перекуплений
        rsi_bear = (rsi_val >= 40) if use_rsi else True # Не перепроданий
        
        # 3. OBV
        obv_val = htf_row.get('obv', 0)
        obv_ma = htf_row.get('obv_ma', 0)
        obv_bull = (obv_val > obv_ma) if use_obv else True
        obv_bear = (obv_val < obv_ma) if use_obv else True

        is_bull = cloud_bull and rsi_bull and obv_bull
        is_bear = cloud_bear and rsi_bear and obv_bear

        return {
            'bull': bool(is_bull), 
            'bear': bool(is_bear), 
            'details': {'rsi': rsi_val}
        }

    def analyze(self, df_ltf, df_htf):
        """Основний метод аналізу для сканера"""
        signals = []
        if df_ltf is None or df_htf is None: return signals
        
        # Підготовка даних
        df_htf = self.calculate_indicators(df_htf)
        df_ltf = self.calculate_indicators(df_ltf)
        
        if len(df_htf) < 2 or len(df_ltf) < 2: return signals

        # 1. Перевірка HTF тренду
        htf_filter = self.check_htf_filters(df_htf.iloc[-1])
        if not htf_filter['bull'] and not htf_filter['bear']:
            return signals

        # 2. Логіка входу на LTF
        curr = df_ltf.iloc[-1]
        prev = df_ltf.iloc[-2]
        
        use_retest = self.get_param('obt_useOBRetest', False)
        
        signal = None
        reason = ""
        sl_price = 0.0

        obs = self.find_order_blocks(df_ltf)

        if htf_filter['bull']:
            # RSI Logic for entry
            rsi_entry = curr['rsi'] < float(self.get_param('obt_entryRsiOversold', 45))
            
            # Retest Logic
            retest_ok = False
            if use_retest and obs['buy']:
                # Ціна торкається OB
                last_ob = obs['buy'][-1]
                if last_ob['bottom'] <= curr['low'] <= last_ob['top']:
                    retest_ok = True
                    sl_price = last_ob['bottom'] * 0.998
            elif not use_retest:
                retest_ok = True # Ігноруємо
                sl_price = curr['low'] * 0.99 # Fallback SL
            
            if retest_ok and rsi_entry:
                signal = "Buy"
                reason = "Trend + RSI + OB" if use_retest else "Trend + RSI"

        elif htf_filter['bear']:
            rsi_entry = curr['rsi'] > float(self.get_param('obt_entryRsiOverbought', 55))
            
            retest_ok = False
            if use_retest and obs['sell']:
                last_ob = obs['sell'][-1]
                if last_ob['bottom'] <= curr['high'] <= last_ob['top']:
                    retest_ok = True
                    sl_price = last_ob['top'] * 1.002
            elif not use_retest:
                retest_ok = True
                sl_price = curr['high'] * 1.01

            if retest_ok and rsi_entry:
                signal = "Sell"
                reason = "Trend + RSI + OB" if use_retest else "Trend + RSI"

        if signal:
            signals.append({
                'action': signal,
                'price': curr['close'],
                'sl_price': sl_price,
                'reason': reason,
                'rsi': curr['rsi']
            })
            
        return signals

    def check_exit_signal(self, df_htf, position_side):
        """Логіка розумного виходу (Smart Exit)"""
        res = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None: return res
        
        try:
            df = self.calculate_indicators(df_htf)
            last = df.iloc[-1]
            
            # Параметри
            rsi_overbought = float(self.get_param('exit_rsiOverbought', 70))
            rsi_oversold = float(self.get_param('exit_rsiOversold', 30))
            
            res['details']['rsi'] = round(last['rsi'], 1)
            
            if position_side == "Buy":
                if last['rsi'] >= rsi_overbought:
                    res['close'] = True
                    res['reason'] = f"RSI Overbought ({round(last['rsi'],1)})"
            elif position_side == "Sell":
                if last['rsi'] <= rsi_oversold:
                    res['close'] = True
                    res['reason'] = f"RSI Oversold ({round(last['rsi'],1)})"
                    
        except Exception as e:
            logger.error(f"Exit Check Error: {e}")
            
        return res

ob_trend_strategy = ObTrendStrategy()