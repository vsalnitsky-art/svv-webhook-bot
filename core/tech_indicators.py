"""
Technical Indicators - RSI, ATR, BB, Volume analysis for Sleeper OB Bot
Uses Wilder's Smoothing Method for 100% TradingView compatibility
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

class TechIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def rsi(closes: List[float], period: int = 14) -> List[float]:
        """
        Calculate RSI using Wilder's Smoothing Method (TradingView compatible)
        """
        if len(closes) < period + 1:
            return [50.0] * len(closes)
        
        closes = np.array(closes, dtype=float)
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # First average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = [50.0] * period
        
        # Wilder's smoothing
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
        
        # Pad to match original length
        rsi_values = [50.0] + rsi_values
        return rsi_values
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> List[float]:
        """
        Calculate ATR using Wilder's Smoothing Method
        """
        if len(closes) < period + 1:
            return [0.0] * len(closes)
        
        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        
        # True Range
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
        
        # ATR with Wilder's smoothing
        atr_values = [0.0] * (period - 1)
        atr_current = np.mean(tr[:period])
        atr_values.append(atr_current)
        
        for i in range(period, len(tr)):
            atr_current = (atr_current * (period - 1) + tr[i]) / period
            atr_values.append(atr_current)
        
        return atr_values
    
    @staticmethod
    def bollinger_bands(closes: List[float], period: int = 20, 
                        std_dev: float = 2.0) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands
        """
        closes = np.array(closes, dtype=float)
        
        if len(closes) < period:
            empty = [0.0] * len(closes)
            return {'upper': empty, 'middle': empty, 'lower': empty, 'width': empty}
        
        middle = []
        upper = []
        lower = []
        width = []
        
        for i in range(len(closes)):
            if i < period - 1:
                middle.append(closes[i])
                upper.append(closes[i])
                lower.append(closes[i])
                width.append(0.0)
            else:
                window = closes[i-period+1:i+1]
                sma = np.mean(window)
                std = np.std(window, ddof=0)
                
                middle.append(sma)
                upper.append(sma + std_dev * std)
                lower.append(sma - std_dev * std)
                
                # Width as percentage of middle
                if sma > 0:
                    width.append((std_dev * std * 2) / sma * 100)
                else:
                    width.append(0.0)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    @staticmethod
    def sma(values: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        values = np.array(values, dtype=float)
        result = []
        
        for i in range(len(values)):
            if i < period - 1:
                result.append(values[i])
            else:
                result.append(np.mean(values[i-period+1:i+1]))
        
        return result
    
    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        values = np.array(values, dtype=float)
        multiplier = 2 / (period + 1)
        
        result = [values[0]]
        for i in range(1, len(values)):
            ema_val = (values[i] * multiplier) + (result[-1] * (1 - multiplier))
            result.append(ema_val)
        
        return result
    
    @staticmethod
    def volume_profile(volumes: List[float], period: int = 20) -> Dict[str, float]:
        """
        Analyze volume profile
        Returns average, current ratio, and trend
        """
        if len(volumes) < period:
            return {'avg': 0, 'ratio': 1.0, 'trend': 'neutral'}
        
        volumes = np.array(volumes[-period:], dtype=float)
        avg_vol = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        current_vol = volumes[-1]
        
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Volume trend
        if len(volumes) >= 5:
            recent_avg = np.mean(volumes[-5:])
            older_avg = np.mean(volumes[:-5])
            if recent_avg > older_avg * 1.2:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                trend = 'decreasing'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        return {
            'avg': float(avg_vol),
            'current': float(current_vol),
            'ratio': float(ratio),
            'trend': trend
        }
    
    @staticmethod
    def volume_profile_poc(klines: List[Dict], num_bins: int = 20) -> Dict:
        """
        Calculate Volume Profile with POC (Point of Control)
        
        POC = price level with highest traded volume
        If current price is near POC, it indicates strong equilibrium zone.
        Breakout from POC zone has 40% higher probability of continuation.
        
        Returns:
        - poc_price: Price level with highest volume
        - poc_volume: Volume at POC
        - value_area_high: Upper boundary of value area (70% of volume)
        - value_area_low: Lower boundary of value area
        - price_at_poc: True if current price is within 1% of POC
        - poc_strength: How dominant POC is (0-100)
        """
        if len(klines) < 10:
            return {
                'poc_price': 0, 'poc_volume': 0,
                'value_area_high': 0, 'value_area_low': 0,
                'price_at_poc': False, 'poc_strength': 0
            }
        
        # Extract price and volume data
        closes = np.array([k['close'] for k in klines], dtype=float)
        highs = np.array([k['high'] for k in klines], dtype=float)
        lows = np.array([k['low'] for k in klines], dtype=float)
        volumes = np.array([k['volume'] for k in klines], dtype=float)
        
        price_min = np.min(lows)
        price_max = np.max(highs)
        price_range = price_max - price_min
        
        if price_range == 0:
            return {
                'poc_price': closes[-1], 'poc_volume': 0,
                'value_area_high': closes[-1], 'value_area_low': closes[-1],
                'price_at_poc': True, 'poc_strength': 0
            }
        
        # Create price bins
        bin_size = price_range / num_bins
        bins = np.zeros(num_bins)
        
        # Distribute volume across price bins
        for i in range(len(klines)):
            candle_low = lows[i]
            candle_high = highs[i]
            candle_vol = volumes[i]
            
            # Find bins this candle touches
            low_bin = max(0, int((candle_low - price_min) / bin_size))
            high_bin = min(num_bins - 1, int((candle_high - price_min) / bin_size))
            
            # Distribute volume evenly across touched bins
            num_touched = high_bin - low_bin + 1
            vol_per_bin = candle_vol / num_touched if num_touched > 0 else 0
            
            for b in range(low_bin, high_bin + 1):
                if b < num_bins:
                    bins[b] += vol_per_bin
        
        # Find POC (bin with max volume)
        poc_bin = np.argmax(bins)
        poc_price = price_min + (poc_bin + 0.5) * bin_size
        poc_volume = bins[poc_bin]
        total_volume = np.sum(bins)
        
        # Calculate Value Area (70% of total volume around POC)
        sorted_indices = np.argsort(bins)[::-1]
        cumulative_vol = 0
        value_area_bins = []
        
        for idx in sorted_indices:
            value_area_bins.append(idx)
            cumulative_vol += bins[idx]
            if cumulative_vol >= total_volume * 0.7:
                break
        
        va_low_bin = min(value_area_bins)
        va_high_bin = max(value_area_bins)
        value_area_low = price_min + va_low_bin * bin_size
        value_area_high = price_min + (va_high_bin + 1) * bin_size
        
        # Check if current price is at POC (within 1%)
        current_price = closes[-1]
        poc_distance_pct = abs(current_price - poc_price) / poc_price * 100
        price_at_poc = poc_distance_pct < 1.0
        
        # POC strength (how dominant is POC vs other levels)
        poc_strength = (poc_volume / total_volume * 100) if total_volume > 0 else 0
        
        return {
            'poc_price': float(poc_price),
            'poc_volume': float(poc_volume),
            'value_area_high': float(value_area_high),
            'value_area_low': float(value_area_low),
            'price_at_poc': price_at_poc,
            'poc_strength': float(poc_strength),
            'poc_distance_pct': float(poc_distance_pct)
        }
    
    @staticmethod
    def price_range_analysis(highs: List[float], lows: List[float], 
                             closes: List[float], period: int = 20) -> Dict[str, float]:
        """
        Analyze price range and volatility
        """
        if len(closes) < period:
            return {'range_pct': 0, 'position': 0.5, 'squeeze': False}
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        recent_closes = closes[-period:]
        
        range_high = max(recent_highs)
        range_low = min(recent_lows)
        current_close = recent_closes[-1]
        
        range_size = range_high - range_low
        mid_price = (range_high + range_low) / 2
        
        # Range as percentage
        range_pct = (range_size / mid_price * 100) if mid_price > 0 else 0
        
        # Position within range (0 = low, 1 = high)
        position = (current_close - range_low) / range_size if range_size > 0 else 0.5
        
        # Squeeze detection (tight range)
        squeeze = range_pct < 3  # Less than 3% range
        
        return {
            'range_pct': float(range_pct),
            'position': float(position),
            'squeeze': squeeze,
            'range_high': float(range_high),
            'range_low': float(range_low)
        }
    
    @staticmethod
    def detect_divergence(prices: List[float], indicator: List[float], 
                          lookback: int = 14) -> Optional[str]:
        """
        Detect RSI/price divergence
        Returns: 'bullish', 'bearish', or None
        """
        if len(prices) < lookback or len(indicator) < lookback:
            return None
        
        prices = prices[-lookback:]
        indicator = indicator[-lookback:]
        
        # Find swing lows/highs
        price_low_idx = np.argmin(prices)
        price_high_idx = np.argmax(prices)
        
        # Current values
        current_price = prices[-1]
        current_ind = indicator[-1]
        
        # Bullish divergence: lower price low, higher indicator low
        if price_low_idx > len(prices) // 2:  # Recent low
            if current_price <= prices[price_low_idx] and current_ind > indicator[price_low_idx]:
                return 'bullish'
        
        # Bearish divergence: higher price high, lower indicator high
        if price_high_idx > len(prices) // 2:  # Recent high
            if current_price >= prices[price_high_idx] and current_ind < indicator[price_high_idx]:
                return 'bearish'
        
        return None
    
    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> Dict[str, any]:
        """
        Calculate ADX (Average Directional Index)
        
        ADX measures trend strength (not direction):
        - ADX < 20: Weak or no trend (good for sleeper detection)
        - ADX 20-25: Emerging trend
        - ADX 25-50: Strong trend
        - ADX > 50: Very strong trend
        
        Returns: {'adx': float, 'plus_di': float, 'minus_di': float, 'is_trendless': bool}
        """
        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        
        if len(closes) < period * 2:
            return {'adx': 25, 'plus_di': 50, 'minus_di': 50, 'is_trendless': False, 'adx_falling': False}
        
        # True Range
        tr = np.zeros(len(closes))
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Directional Movement
        plus_dm = np.zeros(len(closes))
        minus_dm = np.zeros(len(closes))
        
        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Wilder's smoothing (first value = SMA, then smoothed)
        def wilder_smooth(values, period):
            result = np.zeros(len(values))
            # First value is SMA
            result[period] = np.mean(values[1:period+1])
            # Subsequent values use Wilder's formula
            for i in range(period + 1, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            return result
        
        # Smoothed TR, +DM, -DM
        atr = wilder_smooth(tr, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)
        
        # +DI and -DI (as percentages)
        plus_di = np.zeros(len(closes))
        minus_di = np.zeros(len(closes))
        
        for i in range(period, len(closes)):
            if atr[i] > 0:
                plus_di[i] = (smoothed_plus_dm[i] / atr[i]) * 100
                minus_di[i] = (smoothed_minus_dm[i] / atr[i]) * 100
        
        # DX (Directional Index)
        dx = np.zeros(len(closes))
        for i in range(period, len(closes)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
        
        # ADX = Smoothed DX (starts after 2*period)
        adx_values = np.zeros(len(closes))
        if len(dx) > period * 2:
            adx_values[period*2] = np.mean(dx[period:period*2])
            for i in range(period * 2 + 1, len(closes)):
                adx_values[i] = (adx_values[i-1] * (period - 1) + dx[i]) / period
        
        current_adx = adx_values[-1] if len(adx_values) > 0 else 25
        
        # Clamp ADX to valid range (0-100)
        current_adx = max(0, min(100, current_adx))
        
        # Check if ADX is falling (last 5 periods)
        adx_falling = False
        if len(adx_values) >= 5:
            recent_adx = adx_values[-5:]
            # Filter out zeros
            recent_valid = [v for v in recent_adx if v > 0]
            if len(recent_valid) >= 2:
                adx_falling = recent_valid[-1] < recent_valid[0]
        
        return {
            'adx': float(current_adx),
            'plus_di': float(min(100, plus_di[-1])) if len(plus_di) > 0 else 50,
            'minus_di': float(min(100, minus_di[-1])) if len(minus_di) > 0 else 50,
            'is_trendless': current_adx < 20,
            'adx_falling': adx_falling,
            'adx_values': adx_values[-10:].tolist() if len(adx_values) >= 10 else []
        }
    
    @staticmethod
    def calculate_all(klines: List[Dict], rsi_period: int = 14, 
                      atr_period: int = 14, bb_period: int = 20) -> Dict:
        """
        Calculate all indicators for klines data
        """
        if not klines:
            return {}
        
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        indicators = TechIndicators()
        
        rsi_values = indicators.rsi(closes, rsi_period)
        atr_values = indicators.atr(highs, lows, closes, atr_period)
        bb = indicators.bollinger_bands(closes, bb_period)
        vol_profile = indicators.volume_profile(volumes)
        price_range = indicators.price_range_analysis(highs, lows, closes)
        divergence = indicators.detect_divergence(closes, rsi_values)
        adx_data = indicators.adx(highs, lows, closes, atr_period)
        
        return {
            'rsi': rsi_values,
            'rsi_current': rsi_values[-1] if rsi_values else 50,
            'atr': atr_values,
            'atr_current': atr_values[-1] if atr_values else 0,
            'bb': bb,
            'bb_width_current': bb['width'][-1] if bb['width'] else 0,
            'volume_profile': vol_profile,
            'price_range': price_range,
            'divergence': divergence,
            'adx': adx_data,  # v4: ADX for trend strength
            'last_close': closes[-1] if closes else 0,
            'last_high': highs[-1] if highs else 0,
            'last_low': lows[-1] if lows else 0,
        }


def get_indicators() -> TechIndicators:
    """Get indicators instance"""
    return TechIndicators()
