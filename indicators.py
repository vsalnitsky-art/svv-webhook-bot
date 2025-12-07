#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Універсальний файл індикаторів.
Включає базові (RSI, ATR) та професійні (Bollinger, Ichimoku, OBV) розрахунки.
"""
import pandas as pd
import numpy as np

# === БАЗОВІ ФУНКЦІЇ (Необхідні для Scanner / Monitor) ===

def simple_rsi(close_prices, period=14):
    """
    Розрахунок RSI за методом Уайлдера (Wilder's Smoothing).
    Це стандарт для більшості бірж та TradingView.
    """
    try:
        # Переконаємося, що це Series
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
            
        if len(close_prices) < period + 1:
            return 50.0
        
        # Розрахунок різниці цін
        deltas = close_prices.diff()
        
        # Розділяємо на приріст (gains) та падіння (losses)
        gains = deltas.where(deltas > 0, 0.0)
        losses = -deltas.where(deltas < 0, 0.0)
        
        # === ВАЖЛИВО: Wilder's Smoothing ===
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Розрахунок RS
        rs = avg_gains / avg_losses
        
        # Розрахунок RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Заповнення пропусків
        rsi = rsi.fillna(100.0 if (not avg_losses.empty and avg_losses.iloc[-1] == 0) else 50.0)
        
        return float(rsi.iloc[-1])
    except Exception as e:
        return 50.0

def simple_atr(high, low, close, period=14):
    """
    Розрахунок ATR (Average True Range) за методом Уайлдера (RMA).
    """
    try:
        if len(high) < period + 1:
            return float(close.iloc[-1]) * 0.02
        
        # True Range Calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's Smoothing для ATR
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        return float(atr.iloc[-1])
    except:
        return float(close.iloc[-1]) * 0.02

def calculate_sma(prices, period=20):
    """Simple Moving Average"""
    try:
        return float(prices.rolling(window=period).mean().iloc[-1])
    except:
        return float(prices.iloc[-1])

def calculate_ema(prices, period=12):
    """Exponential Moving Average (Wrapper for compatibility)"""
    try:
        return prices.ewm(span=period, adjust=False).mean()
    except:
        return prices

def calculate_momentum(prices, period=10):
    """Momentum"""
    try:
        if len(prices) < period:
            return 0.0
        return float(prices.iloc[-1] - prices.iloc[-period])
    except:
        return 0.0

# === ПРОФЕСІЙНІ ІНДИКАТОРИ (ДЛЯ WHALE STRATEGY) ===

def calculate_bollinger_bands(series, length=20, std_dev=2.0):
    """
    Розрахунок смуг Боллінджера та ширини каналу.
    Повертає: upper, middle, lower, bandwidth
    """
    try:
        middle = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Bandwidth: (Upper - Lower) / Middle
        bandwidth = ((upper - lower) / middle).fillna(0)
        
        return upper, middle, lower, bandwidth
    except:
        return None, None, None, None

def calculate_obv(close, volume):
    """On-Balance Volume (OBV)"""
    try:
        change = np.sign(close.diff()).fillna(0)
        obv = (change * volume).cumsum()
        return obv
    except:
        return pd.Series(dtype=float)

def calculate_ichimoku(high, low, close, t=9, k=26, s=52):
    """Ichimoku Cloud (Tenkan, Kijun, Span A, Span B)"""
    try:
        tenkan = (high.rolling(window=t).max() + low.rolling(window=t).min()) / 2
        kijun = (high.rolling(window=k).max() + low.rolling(window=k).min()) / 2
        
        # Span A (Leading Span A)
        span_a = ((tenkan + kijun) / 2)
        
        # Span B (Leading Span B)
        span_b = ((high.rolling(window=s).max() + low.rolling(window=s).min()) / 2)
        
        return tenkan, kijun, span_a, span_b
    except:
        return None, None, None, None

def calculate_slope(series, window=10):
    """Лінійний нахил (Slope)"""
    try:
        if len(series) < window: return 0.0
        y = series.tail(window).values
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope
    except:
        return 0.0