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
            
            # 4. OBV Exit MA (Для стратегії виходу)
            obv_exit_len = int(self._get_param('exit_obvLength', 10))
            
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
                df['obv_exit_ma'] = ta.ema(df['obv'], length=obv_exit_len) # EMA для виходу (швидша)

        except Exception as e:
            # print(f"Indicator calc error: {e}")
            pass
            
        return df

    def find_order_blocks(self, df):
        # ... (Код пошуку OB без змін - скорочено для економії місця, залишаємо як було)
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            current_low = subset['low'].iloc[i]
            current_high = subset['high'].iloc[i]
            is_swing_low = True
            for j in range(1, swing + 1):
                if subset['low'].iloc[i-j] <= current_low or subset['low'].iloc[i+j] <= current_low:
                    is_swing_low = False; break
            if is_swing_low and subset['close'].iloc[i+1] > subset['high'].iloc[i]:
                obs['buy'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
            is_swing_high = True
            for j in range(1, swing + 1):
                if subset['high'].iloc[i-j] >= current_high or subset['high'].iloc[i+j] >= current_high:
                    is_swing_high = False; break
            if is_swing_high and subset['close'].iloc[i+1] < subset['low'].iloc[i]:
                obs['sell'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}

    # === НОВИЙ МЕТОД: ПЕРЕВІРКА ВИХОДУ ===
    def check_exit_signal(self, df_htf, position_side):
        """
        Перевіряє умови виходу для активної позиції на основі HTF.
        position_side: 'Buy' (Long) або 'Sell' (Short)
        Повертає: {'close': bool, 'reason': str, 'details': dict}
        """
        result = {'close': False, 'reason': '', 'details': {}}
        
        if df_htf is None or len(df_htf) < 50: 
            return result
            
        # Розраховуємо індикатори (включаючи obv_exit_ma)
        df = self.calculate_indicators(df_htf)
        last_row = df.iloc[-1]
        
        # Параметри виходу
        exit_rsi_overbought = float(self._get_param('exit_rsiOverbought', 70))
        exit_rsi_oversold = float(self._get_param('exit_rsiOversold', 30))
        
        curr_rsi = last_row.get('rsi', 50)
        curr_obv = last_row.get('obv', 0)
        curr_obv_ma = last_row.get('obv_exit_ma', 0)
        
        # Заповнюємо деталі для UI
        result['details'] = {
            'rsi': round(curr_rsi, 1),
            'obv_cross': 'UP' if curr_obv > curr_obv_ma else 'DOWN'
        }

        # --- ЛОГІКА ВИХОДУ (LONG) ---
        if position_side == 'Buy':
            # 1. Structural RSI Reversal (Просто RSI дуже високий)
            # В оригіналі: crossunder(rsi, 70). Тут спростимо: якщо RSI > рівня - це вже зона виходу.
            rsi_exit = curr_rsi >= exit_rsi_overbought
            
            # 2. Micro Confluence (RSI перекуплений + OBV падає)
            confluence_exit = (curr_rsi >= exit_rsi_overbought) and (curr_obv < curr_obv_ma)
            
            if rsi_exit or confluence_exit:
                result['close'] = True
                result['reason'] = "RSI/OBV Exhaustion"
                if confluence_exit: result['reason'] = "Confluence Exit"

        # --- ЛОГІКА ВИХОДУ (SHORT) ---
        elif position_side == 'Sell':
            # 1. Structural RSI Reversal
            rsi_exit = curr_rsi <= exit_rsi_oversold
            
            # 2. Micro Confluence
            confluence_exit = (curr_rsi <= exit_rsi_oversold) and (curr_obv > curr_obv_ma)
            
            if rsi_exit or confluence_exit:
                result['close'] = True
                result['reason'] = "RSI/OBV Exhaustion"
                if confluence_exit: result['reason'] = "Confluence Exit"
                
        return result

    def analyze(self, df_ltf, df_htf):
        # ... (Код аналізу входу без змін)
        signals = []
        df_htf = self.calculate_indicators(df_htf)
        df_ltf = self.calculate_indicators(df_ltf)
        if df_htf is None or df_ltf is None or df_ltf.empty: return []
        curr_price = df_ltf['close'].iloc[-1]
        htf_row = df_htf.iloc[-1]
        use_cloud = self._get_param('obt_useCloudFilter', True)
        use_obv = self._get_param('obt_useObvFilter', True)
        is_bull_trend = True; is_bear_trend = True
        if use_cloud:
            if htf_row['hma_fast'] <= htf_row['hma_slow']: is_bull_trend = False
            if htf_row['hma_fast'] >= htf_row['hma_slow']: is_bear_trend = False
        if use_obv and 'obv_ma' in htf_row:
            if htf_row['obv'] <= htf_row['obv_ma']: is_bull_trend = False
            if htf_row['obv'] >= htf_row['obv_ma']: is_bear_trend = False
        if not is_bull_trend and not is_bear_trend: return []
        ltf_row = df_ltf.iloc[-1]
        use_retest = self._get_param('obt_useOBRetest', True)
        signal = None; details = []; sl_price = 0.0
        if is_bull_trend:
            rsi_ok = ltf_row['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45))
            if use_retest:
                obs = self.find_order_blocks(df_ltf)
                in_zone = False; active_ob = None
                for ob in obs['buy']:
                    if ob['bottom'] <= curr_price <= (ob['top'] * 1.003):
                        in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok:
                    signal = 'Buy'; details.append("Trend+OB Retest"); sl_price = active_ob['bottom'] * 0.995 
            elif rsi_ok:
                signal = 'Buy'; details.append("Trend+RSI"); sl_price = curr_price * (1 - float(self._get_param('fixedSL', 1.5))/100)
        elif is_bear_trend:
            rsi_ok = ltf_row['rsi'] >= float(self._get_param('obt_entryRsiOverbought', 55))
            if use_retest:
                obs = self.find_order_blocks(df_ltf)
                in_zone = False; active_ob = None
                for ob in obs['sell']:
                    if (ob['bottom'] * 0.997) <= curr_price <= ob['top']:
                        in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok:
                    signal = 'Sell'; details.append("Trend+OB Retest"); sl_price = active_ob['top'] * 1.005
            elif rsi_ok:
                signal = 'Sell'; details.append("Trend+RSI"); sl_price = curr_price * (1 + float(self._get_param('fixedSL', 1.5))/100)
        if signal:
            signals.append({'action': signal, 'price': curr_price, 'rsi': ltf_row['rsi'], 'reason': ", ".join(details), 'sl_price': sl_price})
        return signals

ob_trend_strategy = OBTrendStrategy()