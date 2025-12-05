import pandas as pd
import logging
from settings_manager import settings

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self): pass
    def get_param(self, key): return settings.get(key)

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            cloud_fast = self.get_param('obt_cloudFastLen')
            cloud_slow = self.get_param('obt_cloudSlowLen')
            rsi_len = self.get_param('obt_rsiLength')
            obv_len = self.get_param('obt_obvEntryLen')
            
            # HMA Cloud
            # df['hma_fast'] = ta.hma(df['close'], length=cloud_fast)
            # df['hma_slow'] = ta.hma(df['close'], length=cloud_slow)
            
            # RSI
            # df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            
            # OBV
            # df['obv'] = ta.obv(df['close'], df['volume'])
            # if 'obv' in df: df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
        return df

    def check_htf_filters(self, htf_row):
        """Перевірка фільтрів старшого таймфрейму"""
        if htf_row is None or htf_row.empty: return {'bull': False, 'bear': False}
        
        use_cloud = self.get_param('obt_useCloudFilter')
        use_rsi = self.get_param('obt_useRsiFilter')
        use_obv = self.get_param('obt_useObvFilter')
        
        try:
            # Cloud Logic
            cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
            cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
            
            # RSI Logic
            rsi_bull = (htf_row['rsi'] <= self.get_param('obt_entryRsiOversold')) if use_rsi else True
            rsi_bear = (htf_row['rsi'] >= self.get_param('obt_entryRsiOverbought')) if use_rsi else True
            
            # OBV Logic
            obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
            obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
            
            return {
                'bull': bool(cloud_bull and obv_bull and rsi_bull),
                'bear': bool(cloud_bear and obv_bear and rsi_bear),
                'details': {'rsi': htf_row.get('rsi', 0)}
            }
        except: return {'bull': False, 'bear': False}

    def detect_order_blocks(self, df):
        """Знаходження невикористаних Order Blocks"""
        if df is None or len(df) < 50: return [], []
        bull_obs, bear_obs = [], []
        swing_len = self.get_param('obt_swingLength')
        
        # Оптимізація: аналізуємо останні 300 свічок
        subset = df.tail(300).reset_index(drop=True)
        
        for i in range(swing_len, len(subset) - swing_len):
            # Swing Low (Potential Bull OB)
            is_swing_low = True
            for j in range(1, swing_len + 1):
                if subset['low'].iloc[i] >= subset['low'].iloc[i - j] or \
                   subset['low'].iloc[i] >= subset['low'].iloc[i + j]:
                    is_swing_low = False; break
            
            # Swing High (Potential Bear OB)
            is_swing_high = True
            for j in range(1, swing_len + 1):
                if subset['high'].iloc[i] <= subset['high'].iloc[i - j] or \
                   subset['high'].iloc[i] <= subset['high'].iloc[i + j]:
                    is_swing_high = False; break

            # BEAR OB LOGIC
            if is_swing_high:
                top = subset['high'].iloc[i]
                # Шукаємо злам структури вниз (BOS)
                for k in range(i + 1, len(subset)):
                    if subset['close'].iloc[k] < subset['low'].iloc[i]: 
                        wave = subset.iloc[i:k]
                        max_idx = wave['high'].idxmax()
                        bear_obs.append({
                            'top': subset['high'].iloc[max_idx], 
                            'bottom': subset['low'].iloc[max_idx],
                            'idx': i
                        })
                        break

            # BULL OB LOGIC
            if is_swing_low:
                btm = subset['low'].iloc[i]
                # Шукаємо злам структури вгору (BOS)
                for k in range(i + 1, len(subset)):
                    if subset['close'].iloc[k] > subset['high'].iloc[i]:
                        wave = subset.iloc[i:k]
                        min_idx = wave['low'].idxmin()
                        bull_obs.append({
                            'top': subset['high'].iloc[min_idx], 
                            'bottom': subset['low'].iloc[min_idx],
                            'idx': i
                        })
                        break
        
        # Фільтруємо "живі" блоки (ціна ще не пробила їх протилежну сторону)
        current_price = subset['close'].iloc[-1]
        active_bull = [ob for ob in bull_obs if current_price > ob['bottom']]
        active_bear = [ob for ob in bear_obs if current_price < ob['top']]
        
        return active_bull[-5:], active_bear[-5:]

    def get_signal(self, df_ltf, df_htf):
        """Основна функція: повертає сигнал + розрахований SL"""
        df_ltf = self.calculate_indicators(df_ltf)
        df_htf = self.calculate_indicators(df_htf)
        
        if 'hma_fast' not in df_htf.columns: return {'action': None}
        
        filters = self.check_htf_filters(df_htf.iloc[-1])
        use_retest = self.get_param('obt_useOBRetest')
        sl_mode = self.get_param('sl_mode')
        buffer_pct = self.get_param('obBufferPercent')
        
        current = df_ltf['close'].iloc[-1]
        signal = {'action': None, 'reason': '', 'sl_price': None, 'price': current}

        active_ob = None

        # === BULLISH SCENARIO ===
        if filters['bull']:
            triggered = False
            bulls, _ = self.detect_order_blocks(df_ltf)
            
            if use_retest:
                # Перевіряємо ретест (ціна торкнулася зони або всередині)
                for ob in reversed(bulls): # Дивимось від найсвіжіших
                    # Ціна нижче топа, але вище дна (всередині OB)
                    # *1.001 - невеликий допуск на вхід перед торканням
                    if ob['bottom'] <= current <= (ob['top'] * 1.001): 
                        active_ob = ob
                        triggered = True
                        signal['reason'] = "Bull Trend + OB Retest"
                        break
            else:
                triggered = True
                signal['reason'] = "Bull Trend Only"
                if bulls: active_ob = bulls[-1] # Беремо останній відомий OB для SL

            if triggered:
                signal['action'] = "Buy"
                # Розрахунок SL
                if sl_mode == 'OB_Extremity' and active_ob:
                    # SL = OB Bottom - Buffer%
                    signal['sl_price'] = active_ob['bottom'] * (1 - buffer_pct/100)
                elif sl_mode == 'OB_Extremity' and not active_ob and use_retest:
                     # Якщо вимагаємо OB SL, а OB немає - скасовуємо вхід
                     signal['action'] = None 

        # === BEARISH SCENARIO ===
        elif filters['bear']:
            triggered = False
            _, bears = self.detect_order_blocks(df_ltf)
            
            if use_retest:
                for ob in reversed(bears):
                    if (ob['bottom'] * 0.999) <= current <= ob['top']:
                        active_ob = ob
                        triggered = True
                        signal['reason'] = "Bear Trend + OB Retest"
                        break
            else:
                triggered = True
                signal['reason'] = "Bear Trend Only"
                if bears: active_ob = bears[-1]

            if triggered:
                signal['action'] = "Sell"
                # Розрахунок SL
                if sl_mode == 'OB_Extremity' and active_ob:
                    # SL = OB Top + Buffer%
                    signal['sl_price'] = active_ob['top'] * (1 + buffer_pct/100)
                elif sl_mode == 'OB_Extremity' and not active_ob and use_retest:
                     signal['action'] = None

        return signal

strategy_engine = StrategyEngine()