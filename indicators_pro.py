#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from indicators import calculate_ema, calculate_bollinger_bands

def calculate_rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Relative Volume (RVOL)
    """
    avg_vol = volume.rolling(window=period).mean()
    rvol = volume / avg_vol.replace(0, 1)
    return rvol.fillna(0)

def calculate_keltner_channels(high, low, close, period=20, multiplier=1.5):
    """
    Keltner Channels (для TTM Squeeze)
    """
    # Basis
    basis = calculate_ema(close, period)
    
    # ATR Calculation (Manual TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=10).mean()
    
    upper = basis + (atr * multiplier)
    lower = basis - (atr * multiplier)
    
    return upper, lower

def check_ttm_squeeze(df):
    """
    Перевіряє Squeeze: BB всередині KC
    """
    # Bollinger Bands
    bb_upper, _, bb_lower, _ = calculate_bollinger_bands(df['close'], length=20, std_dev=2.0)
    
    # Keltner Channels
    kc_upper, kc_lower = calculate_keltner_channels(df['high'], df['low'], df['close'], period=20, multiplier=1.5)
    
    if bb_upper is None or kc_upper is None:
        return pd.Series([False] * len(df))

    # Logic: BB Upper < KC Upper AND BB Lower > KC Lower
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    
    return squeeze_on

def calculate_adx(high, low, close, period=14):
    """
    ADX (Wilder's Smoothing)
    """
    if len(close) < period + 1:
        return pd.Series([0]*len(close)), pd.Series([0]*len(close)), pd.Series([0]*len(close))

    # 1. DM+ / DM-
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # 2. TR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 3. Smoothing
    atr_smooth = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # 4. DI
    plus_di = 100 * (plus_di_smooth / atr_smooth)
    minus_di = 100 * (minus_di_smooth / atr_smooth)
    
    # 5. ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.fillna(0), plus_di, minus_di
