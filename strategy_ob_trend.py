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
        if df is None or len(df) < 50: return df
        try:
            fast_len = int(self._get_param('obt_cloudFastLen', 10))
            slow_len = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
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
            if all(subset['low'].iloc[i-j] >= cl and subset['low'].iloc[i+j] >= cl for j in range(1, swing+1)):
                if subset['close'].iloc[i+1] > ch: 
                    obs['buy'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
            if all(subset['high'].iloc[i-j] <= ch and subset['high'].iloc[i+j] <= ch for j in range(1, swing+1)):
                if subset['close'].iloc[i+1] < cl: 
                    obs['sell'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}

    def check_exit_signal(self, df_htf, position_side):
        res = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None: return res
        df = self.calculate_indicators(df_htf)
        last = df.iloc[-1]
        rsi, obv, obv_ma = last.get('rsi', 50), last.get('obv', 0), last.get('obv_exit_ma', 0)
        res['details'] = {'rsi': round(rsi, 1), 'obv_cross': 'UP' if obv > obv_ma else 'DOWN'}
        
        limit_buy = float(self._get_param('exit_rsiOverbought', 70))
        limit_sell = float(self._get_param('exit_rsiOversold', 30))

        if position_side == 'Buy':
            if rsi >= limit_buy: res.update({'close': True, 'reason': 'RSI Max'})
            elif rsi >= limit_buy and obv < obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        elif position_side == 'Sell':
            if rsi <= limit_sell: res.update({'close': True, 'reason': 'RSI Min'})
            elif rsi <= limit_sell and obv > obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        return res

    def analyze(self, df_ltf, df_htf):
        signals = []
        df_h = self.calculate_indicators(df_htf)
        df_l = self.calculate_indicators(df_ltf)
        
        if df_h is None or df_l is None: return []
        
        row_h = df_h.iloc[-1]
        row_l = df_l.iloc[-1]
        curr_price = row_l['close']
        
        # 1. Визначаємо дозволені напрямки (Trend)
        allow_long = True
        allow_short = True
        
        # Cloud Filter
        if self._get_param('obt_useCloudFilter', True):
            if row_h.get('hma_fast',0) <= row_h.get('hma_slow',0): allow_long = False
            if row_h.get('hma_fast',0) >= row_h.get('hma_slow',0): allow_short = False
            
        # OBV Filter
        if self._get_param('obt_useObvFilter', True):
            if row_h.get('obv',0) <= row_h.get('obv_ma',0): allow_long = False
            if row_h.get('obv',0) >= row_h.get('obv_ma',0): allow_short = False

        # HTF RSI Filter (Загальний тренд)
        if self._get_param('obt_useRsiFilter', True):
            if row_h.get('rsi', 50) > 55: allow_long = False
            if row_h.get('rsi', 50) < 45: allow_short = False

        # 2. Логіка входу (Trigger)
        use_rsi_filter = self._get_param('obt_useRsiFilter', True) # Чи перевіряти RSI для входу?
        use_retest = self._get_param('obt_useOBRetest', False)
        
        rsi_val = row_l.get('rsi', 50)
        
        # --- LONG ---
        if allow_long:
            # Якщо фільтр RSI увімкнено, вимагаємо перепроданність. Якщо ні - дозволяємо будь-який RSI.
            rsi_condition = (rsi_val <= float(self._get_param('obt_entryRsiOversold', 45))) if use_rsi_filter else True
            
            if rsi_condition:
                signal_found = False
                sl = 0.0
                reason = ""
                
                if use_retest:
                    obs = self.find_order_blocks(df_ltf)
                    # Шукаємо, чи ми в зоні блоку
                    for ob in obs['buy']:
                        if ob['bottom'] <= curr_price <= (ob['top'] * 1.003):
                            signal_found = True
                            sl = ob['bottom'] * 0.995
                            reason = "OB Retest"
                            break
                else:
                    # Якщо без ретесту, то просто сигнал по RSI/Тренду
                    # АЛЕ: Якщо use_rsi_filter вимкнено і retest вимкнено - це вхід на кожній свічці.
                    # Щоб уникнути спаму, в "No Filter Mode" вимагаємо хоча б наявності OB.
                    obs = self.find_order_blocks(df_ltf)
                    if obs['buy']: # Якщо є хоч якісь блоки
                        signal_found = True
                        sl = curr_price * 0.985
                        reason = "Trend+OB Exist"
                
                if signal_found:
                    signals.append({'action': 'Buy', 'price': curr_price, 'rsi': rsi_val, 'reason': reason, 'sl_price': sl})

        # --- SHORT ---
        if allow_short:
            rsi_condition = (rsi_val >= float(self._get_param('obt_entryRsiOverbought', 55))) if use_rsi_filter else True
            
            if rsi_condition:
                signal_found = False
                sl = 0.0
                reason = ""
                
                if use_retest:
                    obs = self.find_order_blocks(df_ltf)
                    for ob in obs['sell']:
                        if (ob['bottom'] * 0.997) <= curr_price <= ob['top']:
                            signal_found = True
                            sl = ob['top'] * 1.005
                            reason = "OB Retest"
                            break
                else:
                    obs = self.find_order_blocks(df_ltf)
                    if obs['sell']:
                        signal_found = True
                        sl = curr_price * 1.015
                        reason = "Trend+OB Exist"
                
                if signal_found:
                    signals.append({'action': 'Sell', 'price': curr_price, 'rsi': rsi_val, 'reason': reason, 'sl_price': sl})

        return signals

ob_trend_strategy = OBTrendStrategy()