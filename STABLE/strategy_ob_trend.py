import pandas_ta as ta
import pandas as pd
import numpy as np
from settings_manager import settings

class OBTrendStrategy:
    def __init__(self): pass

    def _get_param(self, key, default=None):
        val = settings.get(key)
        return val if val is not None else default

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            fast = int(self._get_param('obt_cloudFastLen', 10))
            slow = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast)
            df['hma_slow'] = ta.hma(df['close'], length=slow)
            df['rsi'] = ta.rsi(df['close'], length=int(self._get_param('obt_rsiLength', 14)))
            df['obv'] = ta.obv(df['close'], df['volume'])
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=int(self._get_param('obt_obvEntryLen', 20)))
                df['obv_exit_ma'] = ta.ema(df['obv'], length=int(self._get_param('exit_obvLength', 10)))
        except: pass
        return df

    def find_order_blocks(self, df):
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            cl, ch = subset['low'].iloc[i], subset['high'].iloc[i]
            is_l = all(subset['low'].iloc[i-j] >= cl and subset['low'].iloc[i+j] >= cl for j in range(1, swing+1))
            if is_l and subset['close'].iloc[i+1] > ch:
                obs['buy'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
            is_h = all(subset['high'].iloc[i-j] <= ch and subset['high'].iloc[i+j] <= ch for j in range(1, swing+1))
            if is_h and subset['close'].iloc[i+1] < cl:
                obs['sell'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}

    def check_exit_signal(self, df_htf, position_side):
        res = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None or len(df_htf) < 50: return res
        df = self.calculate_indicators(df_htf); last = df.iloc[-1]
        rsi, obv, obv_ma = last.get('rsi', 50), last.get('obv', 0), last.get('obv_exit_ma', 0)
        res['details'] = {'rsi': round(rsi, 1), 'obv_cross': 'UP' if obv > obv_ma else 'DOWN'}
        
        limit_buy = float(self._get_param('exit_rsiOverbought', 70))
        limit_sell = float(self._get_param('exit_rsiOversold', 30))

        if position_side == 'Buy':
            if rsi >= limit_buy: res.update({'close': True, 'reason': 'RSI Max'})
            if rsi >= limit_buy and obv < obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        elif position_side == 'Sell':
            if rsi <= limit_sell: res.update({'close': True, 'reason': 'RSI Min'})
            if rsi <= limit_sell and obv > obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        return res

    def analyze(self, df_ltf, df_htf):
        sigs = []; df_h = self.calculate_indicators(df_htf); df_l = self.calculate_indicators(df_ltf)
        if df_h is None or df_l is None: return []
        row_h = df_h.iloc[-1]; row_l = df_l.iloc[-1]; price = row_l['close']
        
        is_bull = row_h.get('hma_fast',0) > row_h.get('hma_slow',0)
        if self._get_param('obt_useObvFilter', True):
            if row_h.get('obv',0) <= row_h.get('obv_ma',0): is_bull = False

        sig = None; sl = 0
        if is_bull and row_l['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45)):
            if self._get_param('obt_useOBRetest', False):
                for ob in self.find_order_blocks(df_ltf)['buy']:
                    if ob['bottom'] <= price <= ob['top']*1.003: sig='Buy'; sl=ob['bottom']*0.995; break
            else: sig='Buy'; sl=price*0.985
        
        if sig: sigs.append({'action': sig, 'price': price, 'rsi': row_l['rsi'], 'reason': 'Trend+Strategy', 'sl_price': sl})
        return sigs

ob_trend_strategy = OBTrendStrategy()
