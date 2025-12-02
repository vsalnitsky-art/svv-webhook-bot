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
        if df_15m is None or df_15m.empty: return None
        df = df_15m.copy()
        if 'time' in df.columns: df.set_index('time', inplace=True)
        
        conversion = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        rule = f"{target_tf_minutes}T"
        
        try:
            df_resampled = df.resample(rule).agg(conversion).dropna()
            df_resampled.reset_index(inplace=True)
            return df_resampled
        except: return df

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            prefix = "obt_" 
            fast_len = self.get_param(f'{prefix}cloudFastLen')
            slow_len = self.get_param(f'{prefix}cloudSlowLen')
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            
            rsi_len = self.get_param(f'{prefix}rsiLength')
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            
            obv_len = self.get_param(f'{prefix}obvEntryLen')
            df['obv'] = ta.obv(df['close'], df['volume'])
            if 'obv' in df: df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
            
            df['vol_ma'] = ta.sma(df['volume'], length=20)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        except: pass
        return df

    def find_smart_money_blocks(self, df):
        if df is None or len(df) < 50: return [], []
        
        swing_len = self.get_param('obt_swingLength')
        vol_threshold = self.get_param('obt_volumeSpikeThreshold')
        
        bull_obs, bear_obs = [], []
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        vol_mas = df['vol_ma'].values 
        
        for i in range(swing_len, len(df) - swing_len):
            is_swing_low = True
            for j in range(1, swing_len + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]: is_swing_low = False; break
            
            is_swing_high = True
            for j in range(1, swing_len + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]: is_swing_high = False; break
            
            if is_swing_high:
                swing_low_level = lows[i]
                for k in range(i + 1, len(df)):
                    if closes[k] < swing_low_level:
                        if np.isnan(vol_mas[i]) or np.isnan(vol_mas[k]): continue
                        if (volumes[i] > vol_mas[i] * vol_threshold) or (volumes[k] > vol_mas[k] * vol_threshold):
                            bear_obs.append({
                                'top': float(highs[i]), 'bottom': float(lows[i]), 
                                'time': int(df['time'].iloc[i].timestamp())
                            })
                        break
                        
            if is_swing_low:
                swing_high_level = highs[i]
                for k in range(i + 1, len(df)):
                    if closes[k] > swing_high_level:
                        if np.isnan(vol_mas[i]) or np.isnan(vol_mas[k]): continue
                        if (volumes[i] > vol_mas[i] * vol_threshold) or (volumes[k] > vol_mas[k] * vol_threshold):
                            bull_obs.append({
                                'top': float(highs[i]), 'bottom': float(lows[i]), 
                                'time': int(df['time'].iloc[i].timestamp())
                            })
                        break
        
        current_price = closes[-1]
        active_bull = [ob for ob in bull_obs if current_price > ob['bottom']]
        active_bear = [ob for ob in bear_obs if current_price < ob['top']]
        return active_bull[-5:], active_bear[-5:]

    # === НОВИЙ МЕТОД ДЛЯ ВІЗУАЛІЗАЦІЇ ===
    def get_chart_data(self, df_ltf):
        """Готує JSON для графіка"""
        ltf_tf_setting = str(self.get_param("ltfSelection"))
        if ltf_tf_setting == "45": df = self.resample_candles(df_ltf, 45)
        else: df = df_ltf
            
        if df is None or df.empty: return {}
        
        df = self.calculate_indicators(df)
        bulls, bears = self.find_smart_money_blocks(df)
        
        # Формуємо свічки
        candles = []
        hma_fast = []
        hma_slow = []
        
        for i, row in df.iterrows():
            ts = int(row['time'].timestamp())
            candles.append({'time': ts, 'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']})
            if not pd.isna(row['hma_fast']): hma_fast.append({'time': ts, 'value': row['hma_fast']})
            if not pd.isna(row['hma_slow']): hma_slow.append({'time': ts, 'value': row['hma_slow']})
            
        return {
            "candles": candles,
            "hma_fast": hma_fast,
            "hma_slow": hma_slow,
            "bull_obs": bulls,
            "bear_obs": bears
        }

    def analyze(self, df_ltf, df_htf):
        # ... (Код analyze залишається без змін, він використовує ті самі методи)
        ltf_tf_setting = str(self.get_param("ltfSelection"))
        if ltf_tf_setting == "45": df_trade = self.resample_candles(df_ltf, 45)
        else: df_trade = df_ltf
        if df_trade is None or df_trade.empty: return {'action': None}
        df_trade = self.calculate_indicators(df_trade)
        df_htf = self.calculate_indicators(df_htf)
        if 'hma_fast' not in df_htf.columns: return {'action': None}
        htf_row = df_htf.iloc[-1]
        use_cloud = self.get_param('obt_useCloudFilter')
        use_rsi = self.get_param('obt_useRsiFilter')
        use_obv = self.get_param('obt_useObvFilter')
        cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
        cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
        rsi_bull = (htf_row['rsi'] <= self.get_param('obt_entryRsiOversold')) if use_rsi else True
        rsi_bear = (htf_row['rsi'] >= self.get_param('obt_entryRsiOverbought')) if use_rsi else True
        obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
        obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
        is_bull_trend = cloud_bull and obv_bull and rsi_bull
        is_bear_trend = cloud_bear and obv_bear and rsi_bear
        current_price = df_trade['close'].iloc[-1]
        signal = {'action': None, 'reason': '', 'sl_price': None, 'price': current_price}
        use_retest = self.get_param('obt_useOBRetest')
        buffer_pct = self.get_param('obBufferPercent')
        if is_bull_trend:
            bulls, _ = self.find_smart_money_blocks(df_trade)
            active_ob = None
            if use_retest:
                for ob in reversed(bulls):
                    if ob['bottom'] < current_price <= (ob['top'] * 1.001): 
                        active_ob = ob; signal['action'] = "Buy"; signal['reason'] = "Smart Money OB Retest"; break
            else: signal['action'] = "Buy"; signal['reason'] = "Trend Only"; 
            if signal['action'] == "Buy":
                if active_ob: signal['sl_price'] = active_ob['bottom'] * (1 - buffer_pct/100)
                elif use_retest: signal['action'] = None
        elif is_bear_trend:
            _, bears = self.find_smart_money_blocks(df_trade)
            active_ob = None
            if use_retest:
                for ob in reversed(bears):
                    if (ob['bottom'] * 0.999) <= current_price < ob['top']:
                        active_ob = ob; signal['action'] = "Sell"; signal['reason'] = "Smart Money OB Retest"; break
            else: signal['action'] = "Sell"; signal['reason'] = "Trend Only";
            if signal['action'] == "Sell":
                if active_ob: signal['sl_price'] = active_ob['top'] * (1 + buffer_pct/100)
                elif use_retest: signal['action'] = None
        return signal

ob_trend_strategy = OBCloudStrategyEngine()