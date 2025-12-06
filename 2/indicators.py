#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простіші альтернативи для технічних індикаторів без pandas_ta
"""
import pandas as pd

def simple_rsi(close_prices, period=14):
    """
    Простий розрахунок RSI без pandas_ta
    """
    try:
        if len(close_prices) < period + 1:
            return 50.0  # Нейтральне значення
        
        deltas = close_prices.diff()
        gains = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        losses = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    except:
        return 50.0  # Default

def simple_atr(high, low, close, period=14):
    """
    Простий розрахунок ATR (Average True Range) без pandas_ta
    """
    try:
        if len(high) < period + 1:
            return close.iloc[-1] * 0.02  # 2% по умовчанню
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else close.iloc[-1] * 0.02
    except:
        return close.iloc[-1] * 0.02

def calculate_sma(prices, period=20):
    """
    Простий Moving Average
    """
    try:
        return float(prices.rolling(window=period).mean().iloc[-1])
    except:
        return float(prices.iloc[-1])

def calculate_ema(prices, period=12):
    """
    Exponential Moving Average
    """
    try:
        return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])
    except:
        return float(prices.iloc[-1])

def calculate_momentum(prices, period=10):
    """
    Momentum індикатор
    """
    try:
        if len(prices) < period:
            return 0.0
        return float(prices.iloc[-1] - prices.iloc[-period])
    except:
        return 0.0
