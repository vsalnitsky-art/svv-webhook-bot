import pandas as pd
import pandas_ta as ta
import logging
from settings_manager import settings

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self): pass
    def get_param(self, key): return settings.get(key)

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        cloud_fast = self.get_param('cloudFastLen')
        cloud_slow = self.get_param('cloudSlowLen')
        rsi_len = self.get_param('rsiLength')
        obv_len = self.get_param('obvEntryLen')
        df['hma_fast'] = ta.hma(df['close'], length=cloud_fast)
        df['hma_slow'] = ta.hma(df['close'], length=cloud_slow)
        df['rsi'] = ta.rsi(df['close'], length=rsi_len)
        df['obv'] = ta.obv(df['close'], df['volume'])
        if 'obv' in df: df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        return df

    def check_htf_filters(self, htf_row):
        if htf_row is None or htf_row.empty: return {'bull': False, 'bear': False}
        use_cloud = self.get_param('useCloudFilter')
        use_rsi = self.get_param('useRsiFilter')
        use_obv = self.get_param('useObvFilter')
        
        try:
            cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
            cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
            rsi_bull = (htf_row['rsi'] <= self.get_param('entryRsiOversold')) if use_rsi else True
            rsi_bear = (htf_row['rsi'] >= self.get_param('entryRsiOverbought')) if use_rsi else True
            obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
            obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
            
            is_valid_bull = bool(cloud_bull and obv_bull and rsi_bull)
            is_valid_bear = bool(cloud_bear and obv_bear and rsi_bear)
            
            return {'bull': is_valid_bull, 'bear': is_valid_bear, 'details': {'rsi': htf_row['rsi']}}
        except: return {'bull': False, 'bear': False}

    def detect_order_blocks(self, df):
        if df is None or len(df) < 50: return [], []
        bull_obs, bear_obs = [], []
        swing_len = self.get_param('swingLength')
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing_len, len(subset) - swing_len):
            current_idx = i
            is_swing_high = True
            for j in range(1, swing_len + 1):
                if subset['high'].iloc[current_idx] <= subset['high'].iloc[current_idx - j] or \
                   subset['high'].iloc[current_idx] <= subset['high'].iloc[current_idx + j]:
                    is_swing_high = False; break
            is_swing_low = True
            for j in range(1, swing_len + 1):
                if subset['low'].iloc[current_idx] >= subset['low'].iloc[current_idx - j] or \
                   subset['low'].iloc[current_idx] >= subset['low'].iloc[current_idx + j]:
                    is_swing_low = False; break
            if is_swing_high:
                swing_top = subset['high'].iloc[current_idx]
                for k in range(current_idx + 1, len(subset)):
                    if subset['close'].iloc[k] > swing_top:
                        wave = subset.iloc[current_idx:k]
                        if len(wave) > 0:
                            min_idx = wave['low'].idxmin()
                            bull_obs.append({'top': subset['high'].iloc[min_idx], 'bottom': subset['low'].iloc[min_idx]})
                        break
            if is_swing_low:
                swing_btm = subset['low'].iloc[current_idx]
                for k in range(current_idx + 1, len(subset)):
                    if subset['close'].iloc[k] < swing_btm:
                        wave = subset.iloc[current_idx:k]
                        if len(wave) > 0:
                            max_idx = wave['high'].idxmax()
                            bear_obs.append({'top': subset['high'].iloc[max_idx], 'bottom': subset['low'].iloc[max_idx]})
                        break
        current = subset['close'].iloc[-1]
        return [ob for ob in bull_obs if current > ob['bottom']][-5:], [ob for ob in bear_obs if current < ob['top']][-5:]

    def get_signal(self, df_ltf, df_htf):
        df_ltf = self.calculate_indicators(df_ltf)
        df_htf = self.calculate_indicators(df_htf)
        if 'hma_fast' not in df_htf.columns: return {'action': None, 'params': {}, 'reason': "", 'filters': {}}
        
        filters = self.check_htf_filters(df_htf.iloc[-1])
        use_ob_retest = self.get_param('useOBRetest')
        current = df_ltf['close'].iloc[-1]
        signal, reason = None, ""

        if filters['bull']:
            triggered = False
            if use_ob_retest:
                bull_obs, _ = self.detect_order_blocks(df_ltf)
                for ob in bull_obs:
                    if ob['bottom'] <= current <= ob['top']:
                        triggered = True; reason = "Trend Bull + OB Retest"; break
            else:
                triggered = True; reason = "Trend Bull (No Retest)"
            if triggered: signal = "Buy"

        elif filters['bear']:
            triggered = False
            if use_ob_retest:
                _, bear_obs = self.detect_order_blocks(df_ltf)
                for ob in bear_obs:
                    if ob['bottom'] <= current <= ob['top']:
                        triggered = True; reason = "Trend Bear + OB Retest"; break
            else:
                triggered = True; reason = "Trend Bear (No Retest)"
            if triggered: signal = "Sell"
            
        return {'action': signal, 'reason': reason, 'params': {}, 'filters': filters}

strategy_engine = StrategyEngine()