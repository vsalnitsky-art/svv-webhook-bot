#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📡 RSI/MFI SCREENER v1.0
========================
Модуль сканування ринку на основі RSI/MFI фільтрів.
100% відповідність логіці Pine Script індикатора.

Фільтри (AND logic):
- RSI Value Filter
- MFI Trend Filter  
- Momentum Filter
- HMA Cloud Filter (HTF)
- HTF Signal Filter
- Last Signal Match

Автор: SVV Webhook Bot Team
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pybit.unified_trading import HTTP
from indicators import (
    calculate_rsi_series, 
    calculate_hma, 
    calculate_ema,
    calculate_atr_series
)

logger = logging.getLogger(__name__)

# ============================================================================
#                    HTF AUTO-MAPPING (як в Pine Script)
# ============================================================================

HTF_MAPPING = {
    "1": "15",      # 1m → 15m
    "3": "15",      # 3m → 15m
    "5": "60",      # 5m → 1h
    "15": "60",     # 15m → 1h
    "30": "240",    # 30m → 4h
    "60": "240",    # 1h → 4h
    "120": "240",   # 2h → 4h
    "240": "D",     # 4h → 1D
    "360": "D",     # 6h → 1D
    "720": "D",     # 12h → 1D
    "D": "W",       # 1D → 1W
    "W": "M",       # 1W → 1M
}

def get_auto_htf(main_tf: str) -> str:
    """Автоматичне визначення HTF на основі Main TF"""
    return HTF_MAPPING.get(main_tf, "240")


# ============================================================================
#                    MFI CALCULATION (Money Flow Index)
# ============================================================================

def calculate_mfi_series(high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Money Flow Index (MFI) - 100% відповідність TradingView.
    
    MFI = 100 - (100 / (1 + Money Flow Ratio))
    Money Flow Ratio = Positive Money Flow / Negative Money Flow
    
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = Typical Price × Volume
    """
    # Typical Price
    typical_price = (high + low + close) / 3
    
    # Raw Money Flow
    raw_money_flow = typical_price * volume
    
    # Direction (порівняння з попередньою свічкою)
    tp_change = typical_price.diff()
    
    # Positive/Negative Money Flow
    positive_flow = raw_money_flow.where(tp_change > 0, 0)
    negative_flow = raw_money_flow.where(tp_change < 0, 0)
    
    # Sum over period
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    # Money Flow Ratio та MFI
    # Захист від ділення на нуль
    mfr = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi.fillna(50.0)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: int = 20) -> float:
    """MFI останнє значення"""
    try:
        mfi_series = calculate_mfi_series(high, low, close, volume, period)
        return round(float(mfi_series.iloc[-1]), 2)
    except:
        return 50.0


# ============================================================================
#                    RSI/MFI SCREENER CLASS
# ============================================================================

class RSIMFIScreener:
    """
    Сканер ринку на основі RSI/MFI фільтрів.
    
    Повністю відтворює логіку Pine Script індикатора:
    - RSI сигнали (Buy/Sell)
    - MFI Cloud (Bullish/Bearish)
    - Momentum (Rising/Falling)
    - HMA Cloud (на HTF)
    - HTF Signal
    """
    
    def __init__(self, session: HTTP = None, settings: dict = None):
        """
        Args:
            session: Bybit HTTP session
            settings: Налаштування з settings_manager
        """
        self.session = session
        self.settings = settings or {}
        
        # Defaults (якщо settings не передано)
        self._load_defaults()
    
    def _load_defaults(self):
        """Завантажує налаштування з defaults або settings"""
        s = self.settings
        
        # === TIMEFRAMES ===
        self.main_tf = s.get("screener_main_tf", "60")
        self.htf = s.get("screener_htf", "") or get_auto_htf(self.main_tf)
        
        # === VOLUME ===
        self.min_volume = float(s.get("screener_min_volume", 10000000))
        
        # === RSI SETTINGS ===
        self.rsi_length = int(s.get("screener_rsi_length", 14))
        self.oversold = int(s.get("screener_oversold", 30))
        self.overbought = int(s.get("screener_overbought", 70))
        
        # === MFI SETTINGS ===
        self.mfi_length = int(s.get("screener_mfi_length", 20))
        self.fast_mfi_ema = int(s.get("screener_fast_mfi_ema", 5))
        self.slow_mfi_ema = int(s.get("screener_slow_mfi_ema", 13))
        
        # === HMA SETTINGS ===
        self.hma_fast = int(s.get("screener_hma_fast", 10))
        self.hma_slow = int(s.get("screener_hma_slow", 30))
        
        # === SIGNAL SETTINGS ===
        self.min_peak_strength = int(s.get("screener_min_peak_strength", 2))
        self.require_volume = s.get("screener_require_volume", False)
        self.trend_confirmation = s.get("screener_trend_confirmation", False)
        
        # === FILTER LEVELS ===
        self.rsi_filter_overbought = int(s.get("screener_rsi_filter_overbought", 60))
        self.rsi_filter_oversold = int(s.get("screener_rsi_filter_oversold", 40))
        
        # === FILTERS ON/OFF ===
        self.use_rsi_filter = s.get("screener_use_rsi_filter", True)
        self.use_mfi_filter = s.get("screener_use_mfi_filter", True)
        self.use_momentum_filter = s.get("screener_use_momentum_filter", True)
        self.use_cloud_filter = s.get("screener_use_cloud_filter", True)
        self.use_htf_signal_filter = s.get("screener_use_htf_signal_filter", True)
        self.use_last_signal_filter = s.get("screener_use_last_signal_filter", True)
    
    def update_settings(self, settings: dict):
        """Оновлює налаштування"""
        self.settings = settings
        self._load_defaults()
    
    # ========================================================================
    #                    DATA FETCHING
    # ========================================================================
    
    def _fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Отримує свічки з Bybit"""
        try:
            r = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if r.get('retCode') != 0 or not r.get('result', {}).get('list'):
                return None
            
            data = r['result']['list']
            
            # Bybit повертає [timestamp, open, high, low, close, volume, turnover]
            # В зворотньому порядку (новіші спочатку)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Конвертуємо типи
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            
            # Сортуємо від старих до нових (важливо для індикаторів!)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Kline fetch error {symbol}: {e}")
            return None
    
    def _get_all_symbols(self) -> List[Dict]:
        """Отримує всі USDT perpetuals з об'ємом"""
        try:
            r = self.session.get_tickers(category="linear")
            
            if r.get('retCode') != 0:
                return []
            
            symbols = []
            for t in r.get('result', {}).get('list', []):
                symbol = t.get('symbol', '')
                
                # Тільки USDT perpetuals
                if not symbol.endswith('USDT'):
                    continue
                
                # Пропускаємо специфічні токени
                if any(x in symbol for x in ['1000', '10000', 'USDC']):
                    continue
                
                volume_24h = float(t.get('turnover24h', 0) or 0)
                price = float(t.get('lastPrice', 0) or 0)
                change_24h = float(t.get('price24hPcnt', 0) or 0) * 100
                
                symbols.append({
                    'symbol': symbol,
                    'price': price,
                    'volume_24h': volume_24h,
                    'change_24h': change_24h
                })
            
            return symbols
        
        except Exception as e:
            logger.error(f"Get symbols error: {e}")
            return []
    
    # ========================================================================
    #                    INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Розраховує всі індикатори для одного символу.
        
        Returns:
            Dict з усіма індикаторами
        """
        result = {
            'rsi': 50.0,
            'mfi': 50.0,
            'fast_mfi': 50.0,
            'slow_mfi': 50.0,
            'mfi_cloud': 'neutral',
            'momentum': 'neutral',
            'hma_fast': 0,
            'hma_slow': 0,
            'hma_cloud': 'neutral',
            'signal': 'None',
            'signal_strength': 0
        }
        
        if df is None or len(df) < 50:
            return result
        
        try:
            # === RSI ===
            rsi_series = calculate_rsi_series(df['close'], period=self.rsi_length)
            result['rsi'] = round(float(rsi_series.iloc[-1]), 2)
            
            # === MFI ===
            mfi_series = calculate_mfi_series(
                df['high'], df['low'], df['close'], df['volume'],
                period=self.mfi_length
            )
            result['mfi'] = round(float(mfi_series.iloc[-1]), 2)
            
            # MFI Cloud (Fast/Slow EMA)
            fast_mfi = calculate_ema(mfi_series, period=self.fast_mfi_ema)
            slow_mfi = calculate_ema(mfi_series, period=self.slow_mfi_ema)
            
            result['fast_mfi'] = round(float(fast_mfi.iloc[-1]), 2)
            result['slow_mfi'] = round(float(slow_mfi.iloc[-1]), 2)
            
            if result['fast_mfi'] > result['slow_mfi']:
                result['mfi_cloud'] = 'bullish'
            elif result['fast_mfi'] < result['slow_mfi']:
                result['mfi_cloud'] = 'bearish'
            
            # === MOMENTUM ===
            rsi_change = rsi_series.diff(1)
            if len(rsi_change) > 1:
                if rsi_change.iloc[-1] > 0:
                    result['momentum'] = 'rising'
                elif rsi_change.iloc[-1] < 0:
                    result['momentum'] = 'falling'
            
            # === SIGNAL DETECTION ===
            # Peak Detection (як в Pine Script)
            # isPeak = ta.falling(rsi, minPeakStrength) and rsi >= overbought
            # isDip = ta.rising(rsi, minPeakStrength) and rsi <= oversold
            
            rsi_val = result['rsi']
            
            # Falling: RSI падає протягом minPeakStrength свічок
            is_falling = all(rsi_series.iloc[-i] < rsi_series.iloc[-i-1] 
                           for i in range(1, min(self.min_peak_strength + 1, len(rsi_series))))
            
            # Rising: RSI росте протягом minPeakStrength свічок
            is_rising = all(rsi_series.iloc[-i] > rsi_series.iloc[-i-1] 
                          for i in range(1, min(self.min_peak_strength + 1, len(rsi_series))))
            
            is_peak = is_falling and rsi_val >= self.overbought
            is_dip = is_rising and rsi_val <= self.oversold
            
            # Volume та Trend confirmation
            volume_ok = True
            if self.require_volume and len(df) >= 20:
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                volume_ok = df['volume'].iloc[-1] > avg_volume
            
            trend_ok_buy = True
            trend_ok_sell = True
            if self.trend_confirmation and len(df) >= 20:
                ema_20 = calculate_ema(df['close'], 20).iloc[-1]
                trend_ok_buy = df['close'].iloc[-1] > ema_20
                trend_ok_sell = df['close'].iloc[-1] < ema_20
            
            # RSI Change for 2 bars
            rsi_rising_2 = len(rsi_change) > 2 and rsi_change.iloc[-1] > 0 and rsi_change.iloc[-2] > 0
            rsi_falling_2 = len(rsi_change) > 2 and rsi_change.iloc[-1] < 0 and rsi_change.iloc[-2] < 0
            
            # Crossover/Crossunder
            cross_oversold = len(rsi_series) > 1 and rsi_series.iloc[-2] < self.oversold and rsi_series.iloc[-1] >= self.oversold
            cross_overbought = len(rsi_series) > 1 and rsi_series.iloc[-2] > self.overbought and rsi_series.iloc[-1] <= self.overbought
            
            # BUY Signal (як в Pine Script)
            buy_signal = (is_dip and volume_ok and trend_ok_buy and rsi_rising_2) or \
                        (not self.require_volume and not self.trend_confirmation and cross_oversold)
            
            # SELL Signal (як в Pine Script)
            sell_signal = (is_peak and volume_ok and trend_ok_sell and rsi_falling_2) or \
                         (not self.require_volume and not self.trend_confirmation and cross_overbought)
            
            if buy_signal:
                result['signal'] = 'BUY'
                result['signal_strength'] = abs(self.oversold - rsi_val) if rsi_val <= self.oversold else 0
            elif sell_signal:
                result['signal'] = 'SELL'
                result['signal_strength'] = abs(rsi_val - self.overbought) if rsi_val >= self.overbought else 0
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
        
        return result
    
    def _calculate_htf_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Розраховує HTF індикатори (HMA Cloud, HTF Signal).
        """
        result = {
            'hma_fast': 0,
            'hma_slow': 0,
            'hma_cloud': 'neutral',
            'htf_signal': 'None'
        }
        
        if df is None or len(df) < 50:
            return result
        
        try:
            # === HMA CLOUD ===
            hma_fast = calculate_hma(df['close'], period=self.hma_fast)
            hma_slow = calculate_hma(df['close'], period=self.hma_slow)
            
            result['hma_fast'] = round(float(hma_fast.iloc[-1]), 4)
            result['hma_slow'] = round(float(hma_slow.iloc[-1]), 4)
            
            if result['hma_fast'] > result['hma_slow']:
                result['hma_cloud'] = 'bullish'
            elif result['hma_fast'] < result['hma_slow']:
                result['hma_cloud'] = 'bearish'
            
            # === HTF SIGNAL ===
            # Використовуємо ту ж логіку сигналів, що й для Main TF
            rsi_series = calculate_rsi_series(df['close'], period=self.rsi_length)
            rsi_val = float(rsi_series.iloc[-1])
            
            rsi_change = rsi_series.diff(1)
            
            is_falling = all(rsi_series.iloc[-i] < rsi_series.iloc[-i-1] 
                           for i in range(1, min(self.min_peak_strength + 1, len(rsi_series))))
            is_rising = all(rsi_series.iloc[-i] > rsi_series.iloc[-i-1] 
                          for i in range(1, min(self.min_peak_strength + 1, len(rsi_series))))
            
            is_peak = is_falling and rsi_val >= self.overbought
            is_dip = is_rising and rsi_val <= self.oversold
            
            rsi_rising_2 = len(rsi_change) > 2 and rsi_change.iloc[-1] > 0 and rsi_change.iloc[-2] > 0
            rsi_falling_2 = len(rsi_change) > 2 and rsi_change.iloc[-1] < 0 and rsi_change.iloc[-2] < 0
            
            cross_oversold = len(rsi_series) > 1 and rsi_series.iloc[-2] < self.oversold and rsi_series.iloc[-1] >= self.oversold
            cross_overbought = len(rsi_series) > 1 and rsi_series.iloc[-2] > self.overbought and rsi_series.iloc[-1] <= self.overbought
            
            buy_signal = (is_dip and rsi_rising_2) or cross_oversold
            sell_signal = (is_peak and rsi_falling_2) or cross_overbought
            
            if buy_signal:
                result['htf_signal'] = 'BUY'
            elif sell_signal:
                result['htf_signal'] = 'SELL'
            
        except Exception as e:
            logger.error(f"HTF indicator calculation error: {e}")
        
        return result
    
    # ========================================================================
    #                    FILTER LOGIC
    # ========================================================================
    
    def _apply_filters(self, indicators: Dict, htf_indicators: Dict, signal: str) -> Dict:
        """
        Застосовує фільтри (AND logic) - як в Pine Script.
        
        Returns:
            Dict з результатами фільтрів та статусом
        """
        result = {
            'rsi_filter': {'enabled': self.use_rsi_filter, 'passed': False, 'value': ''},
            'mfi_filter': {'enabled': self.use_mfi_filter, 'passed': False, 'value': ''},
            'momentum_filter': {'enabled': self.use_momentum_filter, 'passed': False, 'value': ''},
            'cloud_filter': {'enabled': self.use_cloud_filter, 'passed': False, 'value': ''},
            'htf_signal_filter': {'enabled': self.use_htf_signal_filter, 'passed': False, 'value': ''},
            'all_passed': False
        }
        
        rsi = indicators.get('rsi', 50)
        mfi_cloud = indicators.get('mfi_cloud', 'neutral')
        momentum = indicators.get('momentum', 'neutral')
        hma_cloud = htf_indicators.get('hma_cloud', 'neutral')
        htf_signal = htf_indicators.get('htf_signal', 'None')
        
        # === RSI FILTER ===
        # LONG: RSI ≤ rsiFilterOverbought (60)
        # SHORT: RSI ≥ rsiFilterOversold (40)
        if signal == 'BUY':
            rsi_passed = not self.use_rsi_filter or (rsi <= self.rsi_filter_overbought)
            result['rsi_filter']['value'] = f"{rsi:.1f} ≤ {self.rsi_filter_overbought}"
        elif signal == 'SELL':
            rsi_passed = not self.use_rsi_filter or (rsi >= self.rsi_filter_oversold)
            result['rsi_filter']['value'] = f"{rsi:.1f} ≥ {self.rsi_filter_oversold}"
        else:
            rsi_passed = True
            result['rsi_filter']['value'] = f"{rsi:.1f}"
        
        result['rsi_filter']['passed'] = rsi_passed
        
        # === MFI FILTER ===
        # LONG: bullishCloud (fast > slow)
        # SHORT: bearishCloud (fast < slow)
        if signal == 'BUY':
            mfi_passed = not self.use_mfi_filter or (mfi_cloud == 'bullish')
        elif signal == 'SELL':
            mfi_passed = not self.use_mfi_filter or (mfi_cloud == 'bearish')
        else:
            mfi_passed = True
        
        result['mfi_filter']['passed'] = mfi_passed
        result['mfi_filter']['value'] = mfi_cloud.capitalize()
        
        # === MOMENTUM FILTER ===
        # LONG: rising
        # SHORT: falling
        if signal == 'BUY':
            mom_passed = not self.use_momentum_filter or (momentum == 'rising')
        elif signal == 'SELL':
            mom_passed = not self.use_momentum_filter or (momentum == 'falling')
        else:
            mom_passed = True
        
        result['momentum_filter']['passed'] = mom_passed
        result['momentum_filter']['value'] = '↑' if momentum == 'rising' else '↓' if momentum == 'falling' else '—'
        
        # === HMA CLOUD FILTER (HTF) ===
        # LONG: fast > slow (bullish)
        # SHORT: fast < slow (bearish)
        if signal == 'BUY':
            cloud_passed = not self.use_cloud_filter or (hma_cloud == 'bullish')
        elif signal == 'SELL':
            cloud_passed = not self.use_cloud_filter or (hma_cloud == 'bearish')
        else:
            cloud_passed = True
        
        result['cloud_filter']['passed'] = cloud_passed
        result['cloud_filter']['value'] = hma_cloud.capitalize()
        
        # === HTF SIGNAL FILTER ===
        # LONG: htf_signal == BUY
        # SHORT: htf_signal == SELL
        if signal == 'BUY':
            htf_passed = not self.use_htf_signal_filter or (htf_signal == 'BUY')
        elif signal == 'SELL':
            htf_passed = not self.use_htf_signal_filter or (htf_signal == 'SELL')
        else:
            htf_passed = True
        
        result['htf_signal_filter']['passed'] = htf_passed
        result['htf_signal_filter']['value'] = htf_signal
        
        # === ALL FILTERS PASSED ===
        result['all_passed'] = all([
            result['rsi_filter']['passed'],
            result['mfi_filter']['passed'],
            result['momentum_filter']['passed'],
            result['cloud_filter']['passed'],
            result['htf_signal_filter']['passed']
        ])
        
        return result
    
    # ========================================================================
    #                    MAIN SCAN METHOD
    # ========================================================================
    
    def scan(self, progress_callback=None) -> List[Dict]:
        """
        Виконує повний скан ринку.
        
        Args:
            progress_callback: Функція для оновлення прогресу (current, total)
        
        Returns:
            List[Dict]: Список символів що пройшли всі фільтри
        """
        results = []
        
        try:
            # 1. Отримуємо всі символи
            all_symbols = self._get_all_symbols()
            logger.info(f"Total symbols found: {len(all_symbols)}")
            
            # 2. Фільтруємо по об'єму
            filtered_symbols = [s for s in all_symbols if s['volume_24h'] >= self.min_volume]
            logger.info(f"After volume filter (≥${self.min_volume/1e6:.0f}M): {len(filtered_symbols)}")
            
            total = len(filtered_symbols)
            
            # 3. Скануємо кожен символ
            for idx, sym_data in enumerate(filtered_symbols):
                symbol = sym_data['symbol']
                
                # Progress callback
                if progress_callback:
                    progress_callback(idx + 1, total)
                
                try:
                    # Отримуємо свічки Main TF
                    df_main = self._fetch_klines(symbol, self.main_tf, limit=200)
                    if df_main is None or len(df_main) < 50:
                        continue
                    
                    # Розраховуємо індикатори Main TF
                    indicators = self._calculate_indicators(df_main)
                    
                    # Перевіряємо чи є сигнал
                    signal = indicators.get('signal', 'None')
                    if signal == 'None':
                        continue
                    
                    # Отримуємо свічки HTF
                    df_htf = self._fetch_klines(symbol, self.htf, limit=200)
                    htf_indicators = self._calculate_htf_indicators(df_htf)
                    
                    # Застосовуємо фільтри
                    filters = self._apply_filters(indicators, htf_indicators, signal)
                    
                    # Тільки 100% match
                    if not filters['all_passed']:
                        continue
                    
                    # Формуємо результат
                    result = {
                        'symbol': symbol,
                        'price': sym_data['price'],
                        'volume_24h': sym_data['volume_24h'],
                        'change_24h': sym_data['change_24h'],
                        'rsi': indicators['rsi'],
                        'mfi': indicators['mfi'],
                        'mfi_cloud': indicators['mfi_cloud'],
                        'momentum': indicators['momentum'],
                        'hma_cloud': htf_indicators['hma_cloud'],
                        'htf_signal': htf_indicators['htf_signal'],
                        'signal': signal,
                        'filters': filters,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    results.append(result)
                    logger.info(f"✅ MATCH: {symbol} - {signal}")
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Сортуємо по RSI відстані від нейтрального
            # BUY: менший RSI краще (перепроданість)
            # SELL: більший RSI краще (перекупленість)
            results.sort(key=lambda x: abs(50 - x['rsi']), reverse=True)
            
            logger.info(f"Scan complete. Found {len(results)} matches.")
            return results
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            return []


# ============================================================================
#                    UTILITY FUNCTIONS
# ============================================================================

def format_volume(volume: float) -> str:
    """Форматує об'єм для відображення"""
    if volume >= 1e9:
        return f"${volume/1e9:.1f}B"
    elif volume >= 1e6:
        return f"${volume/1e6:.1f}M"
    elif volume >= 1e3:
        return f"${volume/1e3:.1f}K"
    else:
        return f"${volume:.0f}"


def format_tf(tf: str) -> str:
    """Форматує таймфрейм для відображення"""
    tf_map = {
        "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
        "60": "1H", "120": "2H", "240": "4H", "360": "6H", "720": "12H",
        "D": "1D", "W": "1W", "M": "1M"
    }
    return tf_map.get(tf, tf)
