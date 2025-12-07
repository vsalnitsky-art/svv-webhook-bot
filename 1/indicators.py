#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простіші альтернативи для технічних індикаторів без pandas_ta.
Включає правильну формулу RSI (Wilder's Smoothing) для точності з TradingView/Bybit.
"""
import pandas as pd
import numpy as np

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
        # Використовуємо ewm з alpha=1/period, що математично еквівалентно методу Уайлдера
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Розрахунок RS
        rs = avg_gains / avg_losses
        
        # Розрахунок RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Заповнення пропусків (на випадок ділення на нуль)
        # Якщо losses = 0, RSI = 100
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
    """
    Simple Moving Average
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
    Momentum
    """
    try:
        if len(prices) < period:
            return 0.0
        return float(prices.iloc[-1] - prices.iloc[-period])
    except:
        return 0.0
