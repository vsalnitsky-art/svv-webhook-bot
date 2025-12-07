#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐋 WHALE HUNTER SCANNER - IMPROVED VERSION
Стратегія пошуку аномалій: Squeeze + Real Divergence + Ichimoku + Trend Filter
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import logging
from datetime import datetime
from threading import Thread
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WhaleHunterImproved:
    def __init__(self, timeframe='1h'):
        """Ініціалізація сканера з усіма покращеннями"""
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        
        self.TIMEFRAME = timeframe
        self.LIMIT = 200  # Більше даних для точніших розрахунків
        self.MIN_VOLUME = 5000000
        
        # Параметри стратегії
        self.BB_LENGTH = 20
        self.BB_STD = 2.0
        self.BB_WIDTH_THRESHOLD = 0.12  # Стиснення
        
        # Ichimoku
        self.ICHIMOKU_TENKAN = 9
        self.ICHIMOKU_KIJUN = 26
        self.ICHIMOKU_SENKOU = 52
        
        # ADX для trend
        self.ADX_LENGTH = 14
        self.ADX_THRESHOLD = 25
        
        # Сигнали (зберігаються в пам'яті)
        self.active_signals = []
        self.signal_history = []
        
        logging.info(f"✅ WhaleHunter Improved ініціалізовано. TF: {timeframe}")

    def fetch_markets(self):
        """Отримує ліквідні пари"""
        try:
            markets = self.exchange.load_markets()
            symbols = []
            for symbol, data in markets.items():
                if '/USDT:USDT' in symbol and data['active']:
                    turnover = data['info'].get('turnover24h')
                    if turnover and float(turnover) > self.MIN_VOLUME:
                        symbols.append(symbol)
            
            logging.info(f"📊 Знайдено {len(symbols)} ліквідних пар")
            return symbols
        except Exception as e:
            logging.error(f"❌ Помилка завантаження ринків: {e}")
            return []

    def fetch_data(self, symbol):
        """Отримує OHLCV дані"""
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe=self.TIMEFRAME, limit=self.LIMIT)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"❌ Помилка даних для {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        """Розраховує індикатори"""
        try:
            # Bollinger Bands
            bb = df.ta.bbands(length=self.BB_LENGTH, std=self.BB_STD)
            df = pd.concat([df, bb], axis=1)
            
            # BB Width
            bb_high_col = f'BBU_{self.BB_LENGTH}_{self.BB_STD}'
            bb_low_col = f'BBL_{self.BB_LENGTH}_{self.BB_STD}'
            bb_mid_col = f'BBM_{self.BB_LENGTH}_{self.BB_STD}'
            
            df['bb_width'] = (df[bb_high_col] - df[bb_low_col]) / df[bb_mid_col]
            
            # OBV
            df['obv'] = df.ta.obv()
            
            # ADX (для trend)
            adx = df.ta.adx(length=self.ADX_LENGTH)
            df = pd.concat([df, adx], axis=1)
            
            # Ichimoku
            ichimoku = df.ta.ichimoku(
                tenkan=self.ICHIMOKU_TENKAN,
                kijun=self.ICHIMOKU_KIJUN,
                senkou=self.ICHIMOKU_SENKOU
            )
            df = pd.concat([df, ichimoku[0]], axis=1)
            
            return df
        except Exception as e:
            logging.error(f"❌ Помилка індикаторів: {e}")
            return None

    def detect_real_divergence(self, df, lookback=20):
        """ПОКРАЩЕНЕ: Реальна дивергенція (не просто slope)"""
        recent = df.tail(lookback).copy()
        
        if len(recent) < 10:
            return False, "Недостатньо даних"
        
        # Знаходимо останнім 2 локальні максимуми ціни
        price_peaks = []
        obv_at_peaks = []
        
        for i in range(5, len(recent) - 1):
            if (recent['close'].iloc[i] > recent['close'].iloc[i-1] and
                recent['close'].iloc[i] > recent['close'].iloc[i+1]):
                price_peaks.append({
                    'price': recent['close'].iloc[i],
                    'obv': recent['obv'].iloc[i],
                    'idx': i
                })
        
        if len(price_peaks) < 2:
            return False, "Недостатньо пиків"
        
        # Перевіряємо дивергенцію на останніх 2 пиках
        peak1 = price_peaks[-2]
        peak2 = price_peaks[-1]
        
        # Bearish divergence: ціна вище, OBV нижче
        if peak2['price'] > peak1['price'] and peak2['obv'] < peak1['obv']:
            return True, "Bearish Divergence (Price Higher, OBV Lower)"
        
        # Bullish divergence: ціна нижче, OBV вище
        if peak2['price'] < peak1['price'] and peak2['obv'] > peak1['obv']:
            return True, "Bullish Divergence (Price Lower, OBV Higher)"
        
        return False, "No Real Divergence"

    def get_trend_direction(self, df):
        """ПОКРАЩЕНЕ: Trend filter з ADX"""
        try:
            last = df.iloc[-1]
            
            adx_col = f'ADX_{self.ADX_LENGTH}'
            di_plus_col = f'DMP_{self.ADX_LENGTH}'
            di_minus_col = f'DMN_{self.ADX_LENGTH}'
            
            if adx_col not in df.columns:
                return 'UNKNOWN', 0
            
            adx = last[adx_col]
            di_plus = last[di_plus_col]
            di_minus = last[di_minus_col]
            
            if adx > self.ADX_THRESHOLD:
                if di_plus > di_minus:
                    return 'UPTREND', adx
                else:
                    return 'DOWNTREND', adx
            
            return 'SIDEWAYS', adx
        except Exception as e:
            logging.error(f"❌ Trend error: {e}")
            return 'UNKNOWN', 0

    def check_ichimoku_signal(self, df):
        """ПОКРАЩЕНЕ: Повний Ichimoku аналіз"""
        try:
            if len(df) < 2:
                return False, "Недостатньо даних"
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Кольони Ichimoku
            its_col = f'ITS_{self.ICHIMOKU_TENKAN}'
            iks_col = f'IKS_{self.ICHIMOKU_KIJUN}'
            isa_col = f'ISA_{self.ICHIMOKU_TENKAN}'
            isb_col = f'ISB_{self.ICHIMOKU_KIJUN}'
            
            # Перевіряємо наявність колонок
            if not all(col in df.columns for col in [its_col, iks_col, isa_col, isb_col]):
                return False, "Ichimoku columns missing"
            
            tenkan = last[its_col]
            kijun = last[iks_col]
            span_a = last[isa_col]
            span_b = last[isb_col]
            close = last['close']
            
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            
            signals = []
            
            # Signal 1: Tenkan > Kijun (bullish momentum)
            if tenkan > kijun:
                signals.append("Tenkan > Kijun")
            
            # Signal 2: Price > Cloud (above cloud)
            if close > cloud_top:
                signals.append("Price > Cloud")
            
            # Signal 3: Tenkan crossover
            if len(df) >= 2:
                prev_tenkan = prev[its_col]
                prev_kijun = prev[iks_col]
                if prev_tenkan <= prev_kijun and tenkan > kijun:
                    signals.append("Tenkan Crossover")
            
            if len(signals) >= 2:
                msg = " + ".join(signals)
                return True, f"Ichimoku Signal ({msg})"
            
            return False, "Ichimoku Neutral"
        except Exception as e:
            logging.error(f"❌ Ichimoku error: {e}")
            return False, f"Error: {e}"

    def check_squeeze_and_breakout(self, df, lookback=5):
        """Стиснення + Розширення волатильності"""
        try:
            recent = df.tail(lookback)
            
            if 'bb_width' not in recent.columns:
                return False, False, "No BB Width"
            
            bb_widths = recent['bb_width'].values
            avg_width = np.mean(bb_widths[:-1])
            current_width = bb_widths[-1]
            
            is_squeezed = current_width < self.BB_WIDTH_THRESHOLD
            is_expanding = current_width > avg_width * 1.3
            
            return is_squeezed, is_expanding, f"Width: {current_width:.4f}"
        except Exception as e:
            return False, False, f"Error: {e}"

    def calculate_entry_and_levels(self, df):
        """Розраховує Entry, SL, TP"""
        try:
            last = df.iloc[-1]
            
            # Entry = поточна ціна
            entry = last['close']
            
            # SL = нижче низу BB
            bb_low_col = f'BBL_{self.BB_LENGTH}_{self.BB_STD}'
            if bb_low_col in df.columns:
                stop_loss = last[bb_low_col] * 0.99
            else:
                stop_loss = entry * 0.98
            
            # TP = 2x risk
            risk = entry - stop_loss
            take_profit = entry + (risk * 2)
            
            # R:R Ratio
            rr_ratio = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
            
            return {
                'entry': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'rr_ratio': round(rr_ratio, 2)
            }
        except Exception as e:
            logging.error(f"❌ Levels error: {e}")
            return None

    def analyze_symbol(self, symbol):
        """Аналізує одну пару"""
        df = self.fetch_data(symbol)
        if df is None or len(df) < self.ICHIMOKU_SENKOU:
            return None
        
        df = self.calculate_indicators(df)
        if df is None:
            return None
        
        # ВСІХ перевіркі
        squeezed, expanding, bb_msg = self.check_squeeze_and_breakout(df)
        if not squeezed:
            return None
        
        trend, adx = self.get_trend_direction(df)
        if trend == 'DOWNTREND':
            return None
        
        has_divergence, div_msg = self.detect_real_divergence(df)
        if not has_divergence:
            return None
        
        ichi_valid, ichi_msg = self.check_ichimoku_signal(df)
        if not ichi_valid:
            return None
        
        # Все хорошо - готовимо сигнал
        levels = self.calculate_entry_and_levels(df)
        if not levels:
            return None
        
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': self.TIMEFRAME,
            'price': round(df['close'].iloc[-1], 4),
            'entry': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'take_profit': levels['take_profit'],
            'rr_ratio': levels['rr_ratio'],
            'trend': trend,
            'adx': round(adx, 2),
            'bb_width': round(df['bb_width'].iloc[-1], 4),
            'signals': [
                f"Squeeze ({bb_msg})",
                f"Divergence ({div_msg})",
                f"Ichimoku ({ichi_msg})",
                f"Trend ({trend})"
            ],
            'confidence': 90
        }
        
        return signal

    def scan_market(self):
        """Головне сканування"""
        symbols = self.fetch_markets()
        signals = []
        
        logging.info(f"🔄 Сканування {len(symbols)} пар...")
        
        for idx, symbol in enumerate(symbols):
            try:
                signal = self.analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    logging.info(f"✅ СИГНАЛ: {signal['symbol']} | RR: {signal['rr_ratio']}")
            except Exception as e:
                logging.error(f"❌ Помилка {symbol}: {e}")
            
            # Rate limit
            if (idx + 1) % 10 == 0:
                time.sleep(0.5)
        
        self.active_signals = signals
        self.signal_history.extend(signals)
        
        logging.info(f"✅ Сканування завершено. Знайдено {len(signals)} сигналів")
        return signals

    def get_signals(self):
        """Повертає поточні сигнали"""
        return self.active_signals

    def get_history(self, limit=100):
        """Історія сигналів"""
        return self.signal_history[-limit:]


# Глобальний екземпляр
whale_hunter = None

def init_scanner(timeframe='1h'):
    """Ініціалізація глобального сканера"""
    global whale_hunter
    whale_hunter = WhaleHunterImproved(timeframe=timeframe)
    return whale_hunter

def scan_in_background(interval=300):
    """Фоновое сканирование (кажде 5 хвилин)"""
    global whale_hunter
    if whale_hunter is None:
        whale_hunter = WhaleHunterImproved()
    
    def background_loop():
        while True:
            try:
                whale_hunter.scan_market()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"❌ Background scan error: {e}")
                time.sleep(10)
    
    thread = Thread(target=background_loop, daemon=True)
    thread.start()
    logging.info(f"🔄 Фоновое сканирование запущено (интервал: {interval}s)")

if __name__ == "__main__":
    scanner = init_scanner('1h')
    signals = scanner.scan_market()
    
    if signals:
        for sig in signals:
            print(f"\n🐋 {sig['symbol']}")
            print(f"   Entry: {sig['entry']} | SL: {sig['stop_loss']} | TP: {sig['take_profit']}")
            print(f"   R:R: {sig['rr_ratio']} | Confidence: {sig['confidence']}%")
