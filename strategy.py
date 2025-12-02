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
        try:
            df['hma_fast'] = ta.hma(df['close'], length=self.get_param('cloudFastLen'))
            df['hma_slow'] = ta.hma(df['close'], length=self.get_param('cloudSlowLen'))
            df['rsi'] = ta.rsi(df['close'], length=self.get_param('rsiLength'))
            df['obv'] = ta.obv(df['close'], df['volume'])
            if 'obv' in df: df['obv_ma'] = ta.sma(df['obv'], length=self.get_param('obvEntryLen'))
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        except: pass
        return df
    def check_htf_filters(self, htf_row):
        if htf_row is None or 'hma_fast' not in htf_row.index: return {'bull': False, 'bear': False}
        use_cloud, use_rsi, use_obv = self.get_param('useCloudFilter'), self.get_param('useRsiFilter'), self.get_param('useObvFilter')
        cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
        cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
        rsi_bull = (htf_row['rsi'] <= self.get_param('entryRsiOversold')) if use_rsi else True
        rsi_bear = (htf_row['rsi'] >= self.get_param('entryRsiOverbought')) if use_rsi else True
        obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
        obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
        return {'bull': bool(cloud_bull and obv_bull and rsi_bull), 'bear': bool(cloud_bear and obv_bear and rsi_bear), 'details': {'rsi': htf_row['rsi']}}
    def detect_order_blocks(self, df):
        if df is None or len(df) < 50: return [], []
        bull, bear, swing = [], [], self.get_param('swingLength')
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            cur = i
            is_h = all(subset['high'][cur] > subset['high'][cur+j] and subset['high'][cur] > subset['high'][cur-j] for j in range(1, swing+1))
            is_l = all(subset['low'][cur] < subset['low'][cur+j] and subset['low'][cur] < subset['low'][cur-j] for j in range(1, swing+1))
            if is_h:
                top = subset['high'][cur]
                for k in range(cur+1, len(subset)):
                    if subset['close'][k] > top: 
                        wave = subset.iloc[cur:k]; bull.append({'top': wave['high'].max(), 'bottom': wave['low'].min()}); break
            if is_l:
                btm = subset['low'][cur]
                for k in range(cur+1, len(subset)):
                    if subset['close'][k] < btm: 
                        wave = subset.iloc[cur:k]; bear.append({'top': wave['high'].max(), 'bottom': wave['low'].min()}); break
        curr = subset['close'].iloc[-1]
        return [b for b in bull if curr > b['bottom']][-5:], [b for b in bear if curr < b['top']][-5:]
    def get_signal(self, df_ltf, df_htf):
        if df_ltf is None or 'hma_fast' not in df_htf.columns: return {'action': None, 'reason': ''}
        filters = self.check_htf_filters(df_htf.iloc[-1])
        use_retest = self.get_param('useOBRetest')
        cur = df_ltf['close'].iloc[-1]
        sig, reason = None, ""
        if filters['bull']:
            trig = False
            if use_retest:
                bulls, _ = self.detect_order_blocks(df_ltf)
                for b in bulls: 
                    if b['bottom'] <= cur <= b['top']: trig = True; reason = "Bull Retest"; break
            else: trig = True; reason = "Bull Trend"
            if trig: sig = "Buy"
        elif filters['bear']:
            trig = False
            if use_retest:
                _, bears = self.detect_order_blocks(df_ltf)
                for b in bears:
                    if b['bottom'] <= cur <= b['top']: trig = True; reason = "Bear Retest"; break
            else: trig = True; reason = "Bear Trend"
            if trig: sig = "Sell"
        return {'action': sig, 'reason': reason}
strategy_engine = StrategyEngine()
