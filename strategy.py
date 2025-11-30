import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from settings_manager import settings

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self):
        pass
        
    def get_param(self, key):
        return settings.get(key)

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df

        cloud_fast = self.get_param('cloudFastLen')
        cloud_slow = self.get_param('cloudSlowLen')
        rsi_len = self.get_param('rsiLength')
        mfi_len = self.get_param('mfiLength')
        obv_len = self.get_param('obvEntryLen')
        atr_len = 14
        
        df['hma_fast'] = ta.hma(df['close'], length=cloud_fast)
        df['hma_slow'] = ta.hma(df['close'], length=cloud_slow)
        df['rsi'] = ta.rsi(df['close'], length=rsi_len)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=mfi_len)
        if 'mfi' in df:
            df['mfi_fast'] = ta.ema(df['mfi'], length=5)
            df['mfi_slow'] = ta.ema(df['mfi'], length=13)
        df['obv'] = ta.obv(df['close'], df['volume'])
        if 'obv' in df:
            df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
        return df

    def check_htf_filters(self, htf_row):
        if htf_row is None or htf_row.empty: return {'bull': False, 'bear': False}
        
        use_cloud = self.get_param('useCloudFilter')
        use_rsi = self.get_param('useRsiFilter')
        use_obv = self.get_param('useObvFilter')
        use_mfi = self.get_param('useMfiFilter')
        
        rsi_oversold = self.get_param('entryRsiOversold')
        rsi_overbought = self.get_param('entryRsiOverbought')

        try:
            cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
            cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
            
            rsi_bull = (htf_row['rsi'] <= rsi_oversold) if use_rsi else True
            rsi_bear = (htf_row['rsi'] >= rsi_overbought) if use_rsi else True
            
            obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
            obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
            
            mfi_bull = (htf_row['mfi_fast'] > htf_row['mfi_slow']) if use_mfi else True
            mfi_bear = (htf_row['mfi_fast'] < htf_row['mfi_slow']) if use_mfi else True
            
            return {
                'bull': is_valid_bull := (cloud_bull and obv_bull and rsi_bull and mfi_bull),
                'bear': is_valid_bear := (cloud_bear and obv_bear and rsi_bear and mfi_bear),
                'details': {'rsi': htf_row['rsi'], 'cloud': 'Bull' if cloud_bull else 'Bear'}
            }
        except Exception as e:
            logger.error(f"Error checking HTF filters: {e}")
            return {'bull': False, 'bear': False}

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
                            bull_obs.append({'type': 'Bull', 'top': subset['high'].iloc[min_idx], 'bottom': subset['low'].iloc[min_idx], 'time': k})
                        break

            if is_swing_low:
                swing_btm = subset['low'].iloc[current_idx]
                for k in range(current_idx + 1, len(subset)):
                    if subset['close'].iloc[k] < swing_btm:
                        wave = subset.iloc[current_idx:k]
                        if len(wave) > 0:
                            max_idx = wave['high'].idxmax()
                            bear_obs.append({'type': 'Bear', 'top': subset['high'].iloc[max_idx], 'bottom': subset['low'].iloc[max_idx], 'time': k})
                        break
        
        current_price = subset['close'].iloc[-1]
        return [ob for ob in bull_obs if current_price > ob['bottom']][-5:], [ob for ob in bear_obs if current_price < ob['top']][-5:]

    def get_signal(self, df_ltf, df_htf):
        df_ltf = self.calculate_indicators(df_ltf)
        df_htf = self.calculate_indicators(df_htf)
        
        htf_row = df_htf.iloc[-1]
        filters = self.check_htf_filters(htf_row)
        
        # Параметри
        use_ob_retest = self.get_param('useOBRetest')
        atr_sl_mult = self.get_param('atrMultiplierSL')
        atr_tp_mult = self.get_param('atrMultiplierTP')
        
        current_price = df_ltf['close'].iloc[-1]
        atr = df_ltf['atr'].iloc[-1]
        
        signal = None
        tp_sl = {}
        reason = ""

        # === ЛОГІКА ВХОДУ (ОНОВЛЕНА) ===
        
        if filters['bull']:
            entry_triggered = False
            
            if use_ob_retest:
                # Вхід ТІЛЬКИ якщо ціна в зоні OB
                bull_obs, _ = self.detect_order_blocks(df_ltf)
                for ob in bull_obs:
                    if ob['bottom'] <= current_price <= ob['top']:
                        entry_triggered = True
                        reason = f"Trend Bull + OB Retest (RSI: {filters['details'].get('rsi',0):.1f})"
                        break
            else:
                # Вхід відразу, якщо тренд бичачий
                entry_triggered = True
                reason = f"Trend Bull (No Retest) (RSI: {filters['details'].get('rsi',0):.1f})"

            if entry_triggered:
                signal = "Buy"
                sl_dist = atr * atr_sl_mult
                tp_dist = atr * atr_tp_mult
                tp_sl = {'sl': current_price - sl_dist, 'tp': current_price + tp_dist}

        elif filters['bear']:
            entry_triggered = False
            
            if use_ob_retest:
                _, bear_obs = self.detect_order_blocks(df_ltf)
                for ob in bear_obs:
                    if ob['bottom'] <= current_price <= ob['top']:
                        entry_triggered = True
                        reason = f"Trend Bear + OB Retest (RSI: {filters['details'].get('rsi',0):.1f})"
                        break
            else:
                entry_triggered = True
                reason = f"Trend Bear (No Retest) (RSI: {filters['details'].get('rsi',0):.1f})"

            if entry_triggered:
                signal = "Sell"
                sl_dist = atr * atr_sl_mult
                tp_dist = atr * atr_tp_mult
                tp_sl = {'sl': current_price + sl_dist, 'tp': current_price - tp_dist}
        
        return {'action': signal, 'params': tp_sl, 'reason': reason, 'filters': filters}

strategy_engine = StrategyEngine()