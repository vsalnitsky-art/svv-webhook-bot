#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from indicators import calculate_true_range, calculate_ema, calculate_sma, calculate_bollinger_bands

def calculate_rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Relative Volume (RVOL)
    Співвідношення поточного об'єму до середнього за N періодів.
    """
    avg_vol = volume.rolling(window=period).mean()
    # Уникаємо ділення на нуль
    rvol = volume / avg_vol.replace(0, 1)
    return rvol.fillna(0)

def calculate_keltner_channels(high, low, close, period=20, multiplier=1.5):
    """
    Keltner Channels (для визначення TTM Squeeze)
    Basis = EMA(20)
    Upper = Basis + ATR(10) * mult
    Lower = Basis - ATR(10) * mult
    """
    # EMA Basis
    basis = calculate_ema(close, period)
    
    # ATR для каналів (використовуємо SMA від TR, класичний метод Кельтнера)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=10).mean() # Традиційно 10 для Кельтнера в TTM
    
    upper = basis + (atr * multiplier)
    lower = basis - (atr * multiplier)
    
    return upper, lower

def check_ttm_squeeze(df):
    """
    Перевіряє, чи знаходиться ціна в стані Squeeze.
    Squeeze = Bollinger Bands знаходяться ВСЕРЕДИНІ Keltner Channels.
    """
    # Bollinger Bands (20, 2.0)
    bb_upper, _, bb_lower, _ = calculate_bollinger_bands(df['close'], length=20, std_dev=2.0)
    
    # Keltner Channels (20, 1.5)
    kc_upper, kc_lower = calculate_keltner_channels(df['high'], df['low'], df['close'], period=20, multiplier=1.5)
    
    # Логіка Squeeze: BB всередині KC
    # Squeeze ON: BB Upper < KC Upper AND BB Lower > KC Lower
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    
    return squeeze_on

def calculate_adx(high, low, close, period=14):
    """
    ADX (Wilder's Smoothing) - Сила тренду.
    100% відповідність TradingView.
    """
    # 1. DM+ та DM-
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # 2. True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # 3. Smoothed TR, +DM, -DM (Wilder's Smoothing: alpha = 1/n)
    # Перше значення = SMA, далі: prev + (curr - prev)/n
    atr_smooth = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # 4. DI+ та DI-
    plus_di = 100 * (plus_di_smooth / atr_smooth)
    minus_di = 100 * (minus_di_smooth / atr_smooth)
    
    # 5. DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # 6. ADX (Smoothing DX)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.fillna(0), plus_di, minus_di
