import pandas_ta as ta
import pandas as pd
import numpy as np
from settings_manager import settings

class OBTrendStrategy:
    def __init__(self):
        pass

    def _get_param(self, key, default=None):
        val = settings.get(key)
        return val if val is not None else default

    def calculate_indicators(self, df):
        """Розрахунок HMA Cloud, RSI, OBV для переданого DataFrame"""
        if df is None or len(df) < 50: return df
        
        try:
            # 1. Cloud (HMA)
            fast_len = int(self._get_param('obt_cloudFastLen', 10))
            slow_len = int(self._get_param('obt_cloudSlowLen', 40))
            
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            
            # 2. RSI
            rsi_len = int(self._get_param('obt_rsiLength', 14))
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            
            # 3. OBV + Trend
            df['obv'] = ta.obv(df['close'], df['volume'])
            obv_len = int(self._get_param('obt_obvEntryLen', 20))
            
            # Перевірка, чи розрахувався OBV (інколи буває NaN на початку)
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)

        except Exception as e:
            # Логування помилки можна додати, якщо потрібно
            print(f"Indicator calc error: {e}")
            pass
            
        return df

    def find_order_blocks(self, df):
        """
        Знаходить прості фрактальні Order Blocks (Swing High/Low).
        Повертає словник зі списками зон buy та sell.
        """
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        
        swing = int(self._get_param('obt_swingLength', 5))
        
        # Проходимо по історії (оптимізовано: останні 300 свічок)
        subset = df.tail(300).reset_index(drop=True)
        
        for i in range(swing, len(subset) - swing):
            current_low = subset['low'].iloc[i]
            current_high = subset['high'].iloc[i]
            
            # --- BULLISH OB (Swing Low) ---
            # Мінімум посередині нижчий за сусідні зліва і справа
            is_swing_low = True
            for j in range(1, swing + 1):
                if subset['low'].iloc[i-j] <= current_low or subset['low'].iloc[i+j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Перевірка на злам структури (BOS) або імпульс вгору після свічки
                if subset['close'].iloc[i+1] > subset['high'].iloc[i]:
                    obs['buy'].append({
                        'top': subset['high'].iloc[i],
                        'bottom': subset['low'].iloc[i],
                        'created_at': subset['time'].iloc[i]
                    })

            # --- BEARISH OB (Swing High) ---
            is_swing_high = True
            for j in range(1, swing + 1):
                if subset['high'].iloc[i-j] >= current_high or subset['high'].iloc[i+j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                if subset['close'].iloc[i+1] < subset['low'].iloc[i]:
                    obs['sell'].append({
                        'top': subset['high'].iloc[i],
                        'bottom': subset['low'].iloc[i],
                        'created_at': subset['time'].iloc[i]
                    })
        
        # Повертаємо лише 3 останні актуальні блоки для кожного напрямку
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}

    def analyze(self, df_ltf, df_htf):
        """
        Головна функція аналізу.
        Приймає LTF (Low Timeframe) та HTF (High Timeframe) дані.
        Повертає список сигналів.
        """
        signals = []
        
        # 1. Розрахунок індикаторів
        df_htf = self.calculate_indicators(df_htf)
        df_ltf = self.calculate_indicators(df_ltf)
        
        if df_htf is None or df_ltf is None: return []
        if df_ltf.empty or df_htf.empty: return []

        # Останні значення
        curr_price = df_ltf['close'].iloc[-1]
        
        # --- ФІЛЬТРИ ТРЕНДУ (HTF) ---
        # Беремо останню закриту свічку HTF
        htf_row = df_htf.iloc[-1]
        
        use_cloud = self._get_param('obt_useCloudFilter', True)
        use_obv = self._get_param('obt_useObvFilter', True)
        
        is_bull_trend = True
        is_bear_trend = True
        
        # Cloud Filter
        if use_cloud:
            if htf_row['hma_fast'] <= htf_row['hma_slow']: is_bull_trend = False
            if htf_row['hma_fast'] >= htf_row['hma_slow']: is_bear_trend = False
            
        # OBV Filter
        if use_obv and 'obv_ma' in htf_row:
            if htf_row['obv'] <= htf_row['obv_ma']: is_bull_trend = False
            if htf_row['obv'] >= htf_row['obv_ma']: is_bear_trend = False
            
        # Якщо немає чіткого тренду згідно фільтрів - вихід
        if not is_bull_trend and not is_bear_trend:
            return []

        # --- ЛОГІКА ВХОДУ (LTF) ---
        ltf_row = df_ltf.iloc[-1]
        use_retest = self._get_param('obt_useOBRetest', True)
        
        signal = None
        details = []
        sl_price = 0.0
        
        # LONG SCENARIO
        if is_bull_trend:
            rsi_ok = ltf_row['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45))
            
            if use_retest:
                # Перевірка чи ціна в зоні OB
                obs = self.find_order_blocks(df_ltf)
                in_zone = False
                active_ob = None
                
                for ob in obs['buy']:
                    # Ціна знаходиться в межах OB або трохи вище (buffer)
                    # top * 1.002 = 0.2% допуску зверху
                    if ob['bottom'] <= curr_price <= (ob['top'] * 1.003):
                        in_zone = True
                        active_ob = ob
                        break
                
                if in_zone and rsi_ok:
                    signal = 'Buy'
                    details.append("Trend+OB Retest")
                    # SL трохи нижче блоку
                    sl_price = active_ob['bottom'] * 0.995 
            
            elif rsi_ok:
                signal = 'Buy'
                details.append("Trend+RSI")
                # SL фіксований %
                sl_price = curr_price * (1 - float(self._get_param('fixedSL', 1.5))/100)

        # SHORT SCENARIO
        elif is_bear_trend:
            rsi_ok = ltf_row['rsi'] >= float(self._get_param('obt_entryRsiOverbought', 55))
            
            if use_retest:
                obs = self.find_order_blocks(df_ltf)
                in_zone = False
                active_ob = None
                
                for ob in obs['sell']:
                    if (ob['bottom'] * 0.997) <= curr_price <= ob['top']:
                        in_zone = True
                        active_ob = ob
                        break
                        
                if in_zone and rsi_ok:
                    signal = 'Sell'
                    details.append("Trend+OB Retest")
                    sl_price = active_ob['top'] * 1.005
            
            elif rsi_ok:
                signal = 'Sell'
                details.append("Trend+RSI")
                sl_price = curr_price * (1 + float(self._get_param('fixedSL', 1.5))/100)

        # Формування результату
        if signal:
            signals.append({
                'action': signal,
                'price': curr_price,
                'rsi': ltf_row['rsi'],
                'reason': ", ".join(details),
                'sl_price': sl_price
            })
            
        return signals

# Екземпляр стратегії для імпорту
ob_trend_strategy = OBTrendStrategy()