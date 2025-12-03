import pandas_ta as ta
import pandas as pd
import numpy as np
from settings_manager import settings

class OBTrendStrategy:
    def __init__(self): pass
    def _get_param(self, key, default=None):
        val = settings.get(key); return val if val is not None else default
    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            fast_len = int(self._get_param('obt_cloudFastLen', 10))
            slow_len = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            rsi_len = int(self._get_param('obt_rsiLength', 14))
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            df['obv'] = ta.obv(df['close'], df['volume'])
            obv_len = int(self._get_param('obt_obvEntryLen', 20))
            obv_exit_len = int(self._get_param('exit_obvLength', 10))
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
                df['obv_exit_ma'] = ta.ema(df['obv'], length=obv_exit_len)
        except: pass
        return df
    def find_order_blocks(self, df):
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            current_low = subset['low'].iloc[i]; current_high = subset['high'].iloc[i]
            is_swing_low = True
            for j in range(1, swing + 1):
                if subset['low'].iloc[i-j] <= current_low or subset['low'].iloc[i+j] <= current_low: is_swing_low = False; break
            if is_swing_low and subset['close'].iloc[i+1] > subset['high'].iloc[i]:
                obs['buy'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
            is_swing_high = True
            for j in range(1, swing + 1):
                if subset['high'].iloc[i-j] >= current_high or subset['high'].iloc[i+j] >= current_high: is_swing_high = False; break
            if is_swing_high and subset['close'].iloc[i+1] < subset['low'].iloc[i]:
                obs['sell'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}
    def check_exit_signal(self, df_htf, position_side):
        result = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None or len(df_htf) < 50: return result
        df = self.calculate_indicators(df_htf); last_row = df.iloc[-1]
        exit_rsi_overbought = float(self._get_param('exit_rsiOverbought', 70))
        exit_rsi_oversold = float(self._get_param('exit_rsiOversold', 30))
        curr_rsi = last_row.get('rsi', 50); curr_obv = last_row.get('obv', 0); curr_obv_ma = last_row.get('obv_exit_ma', 0)
        result['details'] = {'rsi': round(curr_rsi, 1), 'obv_cross': 'UP' if curr_obv > curr_obv_ma else 'DOWN'}
        if position_side == 'Buy':
            rsi_exit = curr_rsi >= exit_rsi_overbought
            confluence_exit = (curr_rsi >= exit_rsi_overbought) and (curr_obv < curr_obv_ma)
            if rsi_exit or confluence_exit: result['close'] = True; result['reason'] = "Confluence Exit" if confluence_exit else "RSI Exhaustion"
        elif position_side == 'Sell':
            rsi_exit = curr_rsi <= exit_rsi_oversold
            confluence_exit = (curr_rsi <= exit_rsi_oversold) and (curr_obv > curr_obv_ma)
            if rsi_exit or confluence_exit: result['close'] = True; result['reason'] = "Confluence Exit" if confluence_exit else "RSI Exhaustion"
        return result
    def analyze(self, df_ltf, df_htf):
        signals = []
        df_htf = self.calculate_indicators(df_htf); df_ltf = self.calculate_indicators(df_ltf)
        if df_htf is None or df_ltf is None or df_ltf.empty: return []
        curr_price = df_ltf['close'].iloc[-1]; htf_row = df_htf.iloc[-1]
        use_cloud = self._get_param('obt_useCloudFilter', True); use_obv = self._get_param('obt_useObvFilter', True)
        is_bull = True; is_bear = True
        if use_cloud:
            if htf_row['hma_fast'] <= htf_row['hma_slow']: is_bull = False
            if htf_row['hma_fast'] >= htf_row['hma_slow']: is_bear = False
        if use_obv and 'obv_ma' in htf_row:
            if htf_row['obv'] <= htf_row['obv_ma']: is_bull = False
            if htf_row['obv'] >= htf_row['obv_ma']: is_bear = False
        if not is_bull and not is_bear: return []
        ltf_row = df_ltf.iloc[-1]; use_retest = self._get_param('obt_useOBRetest', False)
        signal = None; details = []; sl_price = 0.0
        if is_bull:
            rsi_ok = ltf_row['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45))
            if use_retest:
                obs = self.find_order_blocks(df_ltf); in_zone = False; active_ob = None
                for ob in obs['buy']:
                    if ob['bottom'] <= curr_price <= (ob['top'] * 1.003): in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok: signal = 'Buy'; details.append("Trend+OB Retest"); sl_price = active_ob['bottom'] * 0.995 
            elif rsi_ok: signal = 'Buy'; details.append("Trend+RSI"); sl_price = curr_price * (1 - float(self._get_param('fixedSL', 1.5))/100)
        elif is_bear:
            rsi_ok = ltf_row['rsi'] >= float(self._get_param('obt_entryRsiOverbought', 55))
            if use_retest:
                obs = self.find_order_blocks(df_ltf); in_zone = False; active_ob = None
                for ob in obs['sell']:
                    if (ob['bottom'] * 0.997) <= curr_price <= ob['top']: in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok: signal = 'Sell'; details.append("Trend+OB Retest"); sl_price = active_ob['top'] * 1.005
            elif rsi_ok: signal = 'Sell'; details.append("Trend+RSI"); sl_price = curr_price * (1 + float(self._get_param('fixedSL', 1.5))/100)
        if signal: signals.append({'action': signal, 'price': curr_price, 'rsi': ltf_row['rsi'], 'reason': ", ".join(details), 'sl_price': sl_price})
        return signals
ob_trend_strategy = OBTrendStrategy()