import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from settings_manager import settings

logger = logging.getLogger(__name__)

class OBCloudStrategyEngine:
    def __init__(self):
        pass

    def get_param(self, key):
        return settings.get(key)

    def resample_candles(self, df_15m, target_tf_minutes=45):
        """
        Перетворює 15-хвилинні свічки у 45-хвилинні (або інші кастомні)
        """
        if df_15m is None or df_15m.empty: return None
        
        # Переконуємось, що індекс це час
        df = df_15m.copy()
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
            
        # Логіка ресемплінгу
        # Open = перший open, High = макс high, Low = мін low, Close = останній close, Volume = сума
        conversion = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Rule string: '45T' for 45 minutes
        rule = f"{target_tf_minutes}T"
        
        df_resampled = df.resample(rule).agg(conversion).dropna()
        df_resampled.reset_index(inplace=True)
        
        return df_resampled

    def calculate_indicators(self, df, is_htf=False):
        """Розрахунок технічних індикаторів"""
        if df is None or len(df) < 50: return df
        
        try:
            # Prefixes for settings
            prefix = "obt_" 
            
            # --- Cloud HMA ---
            fast_len = self.get_param(f'{prefix}cloudFastLen')
            slow_len = self.get_param(f'{prefix}cloudSlowLen')
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            
            # --- RSI ---
            rsi_len = self.get_param(f'{prefix}rsiLength')
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            
            # --- OBV ---
            obv_len = self.get_param(f'{prefix}obvEntryLen')
            df['obv'] = ta.obv(df['close'], df['volume'])
            if 'obv' in df: 
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
                
            # --- ATR ---
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
        except Exception as e:
            logger.error(f"Indicator Error: {e}")
            
        return df

    def find_swings_and_blocks(self, df):
        """
        Ітеративний пошук Order Blocks (Swing High/Low + Break of Structure)
        """
        if df is None or len(df) < 50: return [], []
        
        swing_len = self.get_param('obt_swingLength')
        bull_obs = [] # [{'top':, 'bottom':, 'index':}]
        bear_obs = []
        
        # Перетворюємо в numpy для швидкості (хоча ітерація все одно потрібна)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Ітеруємося по свічках, починаючи з swing_len
        # Щоб знайти свінг, нам потрібно (i - swing_len) ... i ... (i + swing_len)
        # Але в реальному часі ми бачимо свінг тільки коли пройшло swing_len свічок ПІСЛЯ піку.
        
        for i in range(swing_len, len(df) - swing_len):
            # 1. Визначення Swing Point (Pivot)
            
            # Swing Low (Potential Bull OB)
            # Low[i] має бути нижчим за всі зліва і справа
            is_swing_low = True
            for j in range(1, swing_len + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_swing_low = False
                    break
            
            # Swing High (Potential Bear OB)
            is_swing_high = True
            for j in range(1, swing_len + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_swing_high = False
                    break
            
            # 2. Перевірка на BOS (Break of Structure)
            
            if is_swing_high: # Bearish candidate
                # Шукаємо, чи ціна пробила Low цього свінга вниз
                swing_low_level = lows[i]
                
                for k in range(i + 1, len(df)):
                    if closes[k] < swing_low_level: # BOS!
                        # Order Block - це найвища свічка в діапазоні свінга
                        # (Спрощена логіка: беремо саму свічку свінга як зону)
                        bear_obs.append({
                            'top': highs[i],
                            'bottom': lows[i],
                            'index': i,
                            'type': 'Bear'
                        })
                        break
                        
            if is_swing_low: # Bullish candidate
                # Шукаємо, чи ціна пробила High цього свінга вверх
                swing_high_level = highs[i]
                
                for k in range(i + 1, len(df)):
                    if closes[k] > swing_high_level: # BOS!
                        bull_obs.append({
                            'top': highs[i],
                            'bottom': lows[i],
                            'index': i,
                            'type': 'Bull'
                        })
                        break
        
        # Фільтруємо "живі" блоки (ціна зараз не пробила їх протилежну сторону)
        current_price = closes[-1]
        active_bull = [ob for ob in bull_obs if current_price > ob['bottom']]
        active_bear = [ob for ob in bear_obs if current_price < ob['top']]
        
        # Повертаємо по 5 останніх
        return active_bull[-5:], active_bear[-5:]

    def analyze(self, df_ltf, df_htf):
        """
        Головна функція аналізу.
        df_ltf - сирі дані (наприклад 15m), які ми переробимо в 45m
        df_htf - трендові дані (4H)
        """
        # 1. Підготовка даних (Resampling)
        ltf_tf_setting = str(self.get_param("ltfSelection"))
        
        if ltf_tf_setting == "45":
            df_trade = self.resample_candles(df_ltf, 45)
        else:
            df_trade = df_ltf # Використовуємо як є
            
        if df_trade is None or df_trade.empty: return {'action': None}
        
        # 2. Розрахунок індикаторів
        df_trade = self.calculate_indicators(df_trade)
        df_htf = self.calculate_indicators(df_htf)
        
        if 'hma_fast' not in df_htf.columns: return {'action': None}
        
        # 3. Перевірка HTF фільтрів
        htf_row = df_htf.iloc[-1]
        
        use_cloud = self.get_param('obt_useCloudFilter')
        use_rsi = self.get_param('obt_useRsiFilter')
        use_obv = self.get_param('obt_useObvFilter')
        use_btc = self.get_param('obt_useBtcDominance') # Поки ігноруємо логіку, якщо немає даних
        
        # Cloud
        cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
        cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
        
        # RSI
        rsi_bull = (htf_row['rsi'] <= self.get_param('obt_entryRsiOversold')) if use_rsi else True
        rsi_bear = (htf_row['rsi'] >= self.get_param('obt_entryRsiOverbought')) if use_rsi else True
        
        # OBV
        obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
        obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
        
        is_bull_trend = cloud_bull and obv_bull and rsi_bull
        is_bear_trend = cloud_bear and obv_bear and rsi_bear
        
        # 4. Пошук Order Blocks та Сигналів
        current_price = df_trade['close'].iloc[-1]
        signal = {'action': None, 'reason': '', 'sl_price': None, 'price': current_price}
        
        use_retest = self.get_param('obt_useOBRetest')
        buffer_pct = self.get_param('obBufferPercent')
        
        if is_bull_trend:
            bulls, _ = self.find_swings_and_blocks(df_trade)
            active_ob = None
            
            if use_retest:
                # Ціна має бути всередині блоку або торкнутися його
                for ob in reversed(bulls):
                    # Вхід в зону (Ціна <= Top) і ще не вибило (Ціна > Bottom)
                    if ob['bottom'] < current_price <= (ob['top'] * 1.001): 
                        active_ob = ob
                        signal['action'] = "Buy"
                        signal['reason'] = "Trend + OB Retest (45m)"
                        break
            else:
                # Без ретесту - просто тренд
                signal['action'] = "Buy"
                signal['reason'] = "Trend Only"
                if bulls: active_ob = bulls[-1]
            
            # SL Calc
            if signal['action'] == "Buy":
                if active_ob:
                    signal['sl_price'] = active_ob['bottom'] * (1 - buffer_pct/100)
                elif use_retest:
                    signal['action'] = None # Немає OB для SL
                    
        elif is_bear_trend:
            _, bears = self.find_swings_and_blocks(df_trade)
            active_ob = None
            
            if use_retest:
                for ob in reversed(bears):
                    # Вхід в зону (Ціна >= Bottom) і ще не вибило (Ціна < Top)
                    if (ob['bottom'] * 0.999) <= current_price < ob['top']:
                        active_ob = ob
                        signal['action'] = "Sell"
                        signal['reason'] = "Trend + OB Retest (45m)"
                        break
            else:
                signal['action'] = "Sell"
                signal['reason'] = "Trend Only"
                if bears: active_ob = bears[-1]
                
            # SL Calc
            if signal['action'] == "Sell":
                if active_ob:
                    signal['sl_price'] = active_ob['top'] * (1 + buffer_pct/100)
                elif use_retest:
                    signal['action'] = None

        return signal

ob_trend_strategy = OBCloudStrategyEngine()