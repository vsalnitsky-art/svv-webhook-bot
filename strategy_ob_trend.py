import pandas as pd
import pandas_ta as ta
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from settings_manager import settings

# === CONFIG HELPER ===
class StrategyConfigAdapter:
    """
    Адаптер конфігурації. 
    Зв'язує логіку стратегії з конкретними ключами (obt_*) в базі налаштувань.
    """
    # Фільтри
    @property
    def useCloudFilter(self): return settings.get("obt_useCloudFilter")
    @property
    def useObvFilter(self): return settings.get("obt_useObvFilter")
    @property
    def useRsiFilter(self): return settings.get("obt_useRsiFilter")
    @property
    def useBtcDominanceEntryFilter(self): return settings.get("obt_useBtcDominance")
    
    # Технічні
    @property
    def cloudFastLen(self): return settings.get("obt_cloudFastLen")
    @property
    def cloudSlowLen(self): return settings.get("obt_cloudSlowLen")
    @property
    def entryRsiOversold(self): return settings.get("obt_entryRsiOversold")
    @property
    def entryRsiOverbought(self): return settings.get("obt_entryRsiOverbought")
    @property
    def rsiLength(self): return settings.get("obt_rsiLength")
    @property
    def obvEntryLenInput(self): return settings.get("obt_obvEntryLen")
    @property
    def obvEntryType(self): return settings.get("obt_obvEntryType")
    @property
    def swingLength(self): return settings.get("obt_swingLength")
    @property
    def maxATRMult(self): return settings.get("obt_maxATRMult")
    
    # Ризик (АВТОНОМНІ)
    @property
    def botRisk(self): return settings.get("obt_riskPercent")
    @property
    def botLev(self): return settings.get("obt_leverage")
    @property
    def fixedTP(self): return settings.get("obt_fixedTP")
    @property
    def fixedSL(self): return settings.get("obt_fixedSL")
    @property
    def tpMode(self): return settings.get("obt_tp_mode")
    @property
    def slMode(self): return settings.get("obt_sl_mode")
    @property
    def obBufferPercent(self): return settings.get("obt_obBufferPercent")
    
    @property
    def atrLength(self): return 14 
    @property
    def obEndMethod(self): return "Wick"

# === DATA STRUCTURES ===
@dataclass
class OrderBlockInfo:
    top: float
    bottom: float
    obVolume: float
    obType: str 
    startTime: pd.Timestamp 
    startIndex: int 
    breaker: bool = False
    retestCount: int = 0
    rsiConfirmed: bool = False
    obvConfirmed: bool = False
    pocLevel: float = np.nan
    optimalEntryPrice: float = np.nan
    strengthScore: float = 0.0

@dataclass
class OBSwing:
    index: int = -1
    price: float = np.nan
    volume: float = np.nan
    crossed: bool = False

# === STRATEGY LOGIC ===
class OBTrendStrategy:
    def __init__(self):
        self.cfg = StrategyConfigAdapter()

    def prepare_data(self, df_ltf, df_htf):
        if df_ltf is None or df_htf is None or len(df_ltf) < 50: return None
        df_ltf = df_ltf.copy(); df_htf = df_htf.copy()

        df_ltf['ATR'] = ta.atr(df_ltf['high'], df_ltf['low'], df_ltf['close'], length=self.cfg.atrLength)
        df_ltf['VolMA'] = ta.sma(df_ltf['volume'], length=20)

        df_htf['HTF_HMA_Fast'] = ta.hma(df_htf['close'], length=self.cfg.cloudFastLen)
        df_htf['HTF_HMA_Slow'] = ta.hma(df_htf['close'], length=self.cfg.cloudSlowLen)
        df_htf['HTF_RSI'] = ta.rsi(df_htf['close'], length=self.cfg.rsiLength)
        
        df_htf['HTF_OBV'] = ta.obv(df_htf['close'], df_htf['volume'])
        if self.cfg.obvEntryType == "SMA": df_htf['HTF_OBV_EntryMA'] = ta.sma(df_htf['HTF_OBV'], length=self.cfg.obvEntryLenInput)
        else: df_htf['HTF_OBV_EntryMA'] = ta.ema(df_htf['HTF_OBV'], length=self.cfg.obvEntryLenInput)

        df_ltf = df_ltf.sort_values('time'); df_htf = df_htf.sort_values('time')
        merged = pd.merge_asof(df_ltf, df_htf[['time', 'HTF_HMA_Fast', 'HTF_HMA_Slow', 'HTF_RSI', 'HTF_OBV', 'HTF_OBV_EntryMA']], on='time', direction='backward')
        return merged.ffill().bfill()

    def check_filters(self, row, direction):
        if self.cfg.useCloudFilter:
            if direction == "Bull" and not (row['HTF_HMA_Fast'] > row['HTF_HMA_Slow']): return False
            if direction == "Bear" and not (row['HTF_HMA_Fast'] < row['HTF_HMA_Slow']): return False
        if self.cfg.useObvFilter:
            if direction == "Bull" and not (row['HTF_OBV'] > row['HTF_OBV_EntryMA']): return False
            if direction == "Bear" and not (row['HTF_OBV'] < row['HTF_OBV_EntryMA']): return False
        if self.cfg.useRsiFilter:
            if direction == "Bull" and row['HTF_RSI'] > self.cfg.entryRsiOversold: return False
            if direction == "Bear" and row['HTF_RSI'] < self.cfg.entryRsiOverbought: return False
        return True

    def analyze(self, df_ltf, df_htf):
        df = self.prepare_data(df_ltf, df_htf)
        if df is None: return []

        length = self.cfg.swingLength
        bullish_obs, bearish_obs, results = [], [], []
        start_scan = max(length + 1, len(df) - 200)
        last_swing_high, last_swing_low = OBSwing(), OBSwing()

        for i in range(start_scan, len(df)):
            current_close = df['close'].iloc[i]
            idx_swing = i - length
            window_high = df['high'].iloc[i-length : i+1]
            window_low = df['low'].iloc[i-length : i+1]
            
            is_pivot_high = df['high'].iloc[idx_swing] == window_high.max()
            is_pivot_low = df['low'].iloc[idx_swing] == window_low.min()
            
            if is_pivot_high: last_swing_high = OBSwing(idx_swing, df['high'].iloc[idx_swing], df['volume'].iloc[idx_swing])
            if last_swing_high.index != -1 and not last_swing_high.crossed:
                if current_close > last_swing_high.price:
                    last_swing_high.crossed = True
                    range_slice = df.iloc[last_swing_high.index : i]
                    min_idx = range_slice['low'].idxmin()
                    box_btm, box_top = df['low'][min_idx], df['high'][min_idx]
                    
                    atr = df['ATR'].iloc[i]
                    if atr > 0 and abs(box_top - box_btm) <= atr * self.cfg.maxATRMult:
                        new_ob = OrderBlockInfo(top=box_top, bottom=box_btm, obVolume=df['volume'][min_idx], obType="Bull", startTime=df['time'][min_idx], startIndex=min_idx)
                        bullish_obs.append(new_ob)
                        if self.check_filters(df.iloc[i], "Bull"):
                            # Розрахунок SL/TP для звіту
                            sl = box_btm * (1 - self.cfg.obBufferPercent/100) if self.cfg.slMode == "OB_Level" else box_top * (1 - self.cfg.fixedSL/100)
                            tp = box_top * (1 + self.cfg.fixedTP/100)
                            results.append({'type': 'Bull', 'price': box_top, 'time': df['time'].iloc[i], 'rsi': df['HTF_RSI'].iloc[i], 'sl': sl, 'tp': tp})

            if is_pivot_low: last_swing_low = OBSwing(idx_swing, df['low'].iloc[idx_swing], df['volume'].iloc[idx_swing])
            if last_swing_low.index != -1 and not last_swing_low.crossed:
                if current_close < last_swing_low.price:
                    last_swing_low.crossed = True
                    range_slice = df.iloc[last_swing_low.index : i]
                    max_idx = range_slice['high'].idxmax()
                    box_top, box_btm = df['high'][max_idx], df['low'][max_idx]
                    
                    atr = df['ATR'].iloc[i]
                    if atr > 0 and abs(box_top - box_btm) <= atr * self.cfg.maxATRMult:
                        new_ob = OrderBlockInfo(top=box_top, bottom=box_btm, obVolume=df['volume'][max_idx], obType="Bear", startTime=df['time'][max_idx], startIndex=max_idx)
                        bearish_obs.append(new_ob)
                        if self.check_filters(df.iloc[i], "Bear"):
                            sl = box_top * (1 + self.cfg.obBufferPercent/100) if self.cfg.slMode == "OB_Level" else box_btm * (1 + self.cfg.fixedSL/100)
                            tp = box_btm * (1 - self.cfg.fixedTP/100)
                            results.append({'type': 'Bear', 'price': box_btm, 'time': df['time'].iloc[i], 'rsi': df['HTF_RSI'].iloc[i], 'sl': sl, 'tp': tp})
        return results

ob_trend_strategy = OBTrendStrategy()