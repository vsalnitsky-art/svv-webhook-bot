"""
RSI Signals Buy/Sell + MFI Cloud 3.0 [BOT]
Перевод из Pine Script в Python с сохранением всей логики

Требуемые библиотеки:
pip install pandas numpy ta-lib
или
pip install pandas numpy pandas-ta
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any


class RSIMFIIndicator:
    """
    Индикатор RSI с сигналами покупки/продажи и MFI облаком.
    Полная реализация логики из Pine Script версии 3.0
    """
    
    def __init__(self,
                 # RSI настройки
                 rsi_length: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 
                 # MFI настройки
                 mfi_length: int = 20,
                 fast_mfi_ema: int = 5,
                 slow_mfi_ema: int = 13,
                 cloud_opacity: int = 40,
                 
                 # Настройки сигналов
                 require_volume: bool = False,
                 trend_confirmation: bool = False,
                 min_peak_strength: int = 2,
                 
                 # Визуальные настройки
                 show_signals: bool = True,
                 show_bullish_signals: bool = True,
                 show_bearish_signals: bool = True,
                 
                 # Настройки бота
                 bot_risk: float = 5.0,
                 bot_leverage: int = 20,
                 bot_tp: float = 0.5,
                 bot_sl: float = 0.5,
                 
                 # Алерты
                 enable_alerts: bool = True):
        
        # RSI параметры
        self.rsi_length = rsi_length
        self.oversold = oversold
        self.overbought = overbought
        
        # MFI параметры
        self.mfi_length = mfi_length
        self.fast_mfi_ema = fast_mfi_ema
        self.slow_mfi_ema = slow_mfi_ema
        self.cloud_opacity = cloud_opacity
        
        # Фильтры сигналов
        self.require_volume = require_volume
        self.trend_confirmation = trend_confirmation
        self.min_peak_strength = min_peak_strength
        
        # Визуальные настройки
        self.show_signals = show_signals
        self.show_bullish_signals = show_bullish_signals
        self.show_bearish_signals = show_bearish_signals
        
        # Настройки бота
        self.bot_risk = bot_risk
        self.bot_leverage = bot_leverage
        self.bot_tp = bot_tp
        self.bot_sl = bot_sl
        
        # Алерты
        self.enable_alerts = enable_alerts
        
        # Состояние последних сигналов
        self.last_signal = "Нет"
        self.last_signal_color = "gray"
        self.last_strong_buy_signal = False
        self.last_strong_sell_signal = False
        self.last_regular_buy_signal = False
        self.last_regular_sell_signal = False
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Расчет RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Расчет EMA (Exponential Moving Average)"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Расчет MFI (Money Flow Index)
        Требуется: high, low, close, volume
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi
    
    def detect_peak(self, series: pd.Series, strength: int, threshold: float, 
                    direction: str = 'falling') -> pd.Series:
        """
        Обнаружение пиков/впадин
        direction: 'falling' для пиков, 'rising' для впадин
        """
        result = pd.Series(False, index=series.index)
        
        for i in range(strength, len(series)):
            if direction == 'falling':
                # Проверяем, что RSI падал последние strength баров
                is_falling = all(series.iloc[i-j] > series.iloc[i-j+1] 
                               for j in range(1, strength + 1))
                # И что strength баров назад было выше threshold
                if is_falling and series.iloc[i-strength] >= threshold:
                    result.iloc[i] = True
            
            elif direction == 'rising':
                # Проверяем, что RSI рос последние strength баров
                is_rising = all(series.iloc[i-j] < series.iloc[i-j+1] 
                              for j in range(1, strength + 1))
                # И что strength баров назад было ниже threshold
                if is_rising and series.iloc[i-strength] <= threshold:
                    result.iloc[i] = True
        
        return result
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Основной метод расчета всех индикаторов
        
        Требуемые колонки в df: open, high, low, close, volume
        """
        df = df.copy()
        
        # Расчет RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_length)
        df['smoothed_rsi'] = self.calculate_ema(df['rsi'], 3)
        
        # Расчет MFI
        df['mfi'] = self.calculate_mfi(df, self.mfi_length)
        df['fast_mfi'] = self.calculate_ema(df['mfi'], self.fast_mfi_ema)
        df['slow_mfi'] = self.calculate_ema(df['mfi'], self.slow_mfi_ema)
        
        # Условия облака
        df['bullish_cloud'] = df['fast_mfi'] > df['slow_mfi']
        df['bearish_cloud'] = df['fast_mfi'] < df['slow_mfi']
        
        # Обнаружение пиков и впадин
        df['is_peak'] = self.detect_peak(df['rsi'], self.min_peak_strength, 
                                         self.overbought, 'falling')
        df['is_dip'] = self.detect_peak(df['rsi'], self.min_peak_strength, 
                                        self.oversold, 'rising')
        
        # Фильтр объема
        if self.require_volume:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ok'] = df['volume'] > df['volume_sma']
        else:
            df['volume_ok'] = True
        
        # Фильтр тренда
        if self.trend_confirmation:
            df['ema_20'] = self.calculate_ema(df['close'], 20)
            df['trend_ok_buy'] = df['close'] > df['ema_20']
            df['trend_ok_sell'] = df['close'] < df['ema_20']
        else:
            df['trend_ok_buy'] = True
            df['trend_ok_sell'] = True
        
        # Пересечения уровней
        df['cross_oversold'] = (df['rsi'] > self.oversold) & (df['rsi'].shift(1) <= self.oversold)
        df['cross_overbought'] = (df['rsi'] < self.overbought) & (df['rsi'].shift(1) >= self.overbought)
        
        # Изменение тренда MFI
        df['mfi_bullish_alert'] = df['bullish_cloud'] & (df['bullish_cloud'].shift(1) == False)
        df['mfi_bearish_alert'] = df['bearish_cloud'] & (df['bearish_cloud'].shift(1) == False)
        
        # Расчет изменений RSI
        df['rsi_change_1'] = df['rsi'].diff(1)
        df['rsi_change_2'] = df['rsi'].diff(2)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация торговых сигналов
        """
        df = df.copy()
        
        # Улучшенный сигнал покупки
        buy_condition_1 = (df['is_dip'] & 
                          df['volume_ok'] & 
                          df['trend_ok_buy'] & 
                          (df['rsi_change_1'] > 0) & 
                          (df['rsi_change_2'] > 0))
        
        buy_condition_2 = ((not self.require_volume and 
                           not self.trend_confirmation) & 
                          df['cross_oversold'])
        
        df['buy_signal'] = self.show_signals and (buy_condition_1 | buy_condition_2)
        
        # Улучшенный сигнал продажи
        sell_condition_1 = (df['is_peak'] & 
                           df['volume_ok'] & 
                           df['trend_ok_sell'] & 
                           (df['rsi_change_1'] < 0) & 
                           (df['rsi_change_2'] < 0))
        
        sell_condition_2 = ((not self.require_volume and 
                            not self.trend_confirmation) & 
                           df['cross_overbought'])
        
        df['sell_signal'] = self.show_signals and (sell_condition_1 | sell_condition_2)
        
        # Бычьи сигналы
        df['bullish_cross_over'] = self.show_bullish_signals & df['cross_oversold']
        df['bullish_mfi_signal'] = self.show_bullish_signals & df['mfi_bullish_alert']
        df['strong_bullish_signal'] = (self.show_bullish_signals & 
                                       df['buy_signal'] & 
                                       df['bullish_cloud'])
        
        # Медвежьи сигналы
        df['bearish_cross_under'] = self.show_bearish_signals & df['cross_overbought']
        df['bearish_mfi_signal'] = self.show_bearish_signals & df['mfi_bearish_alert']
        df['strong_bearish_signal'] = (self.show_bearish_signals & 
                                       df['sell_signal'] & 
                                       df['bearish_cloud'])
        
        # Определение последнего сигнала
        df['last_signal'] = 'Нет'
        df.loc[df['buy_signal'], 'last_signal'] = 'ПОКУПКА'
        df.loc[df['sell_signal'], 'last_signal'] = 'ПРОДАЖА'
        
        # Статус RSI
        df['rsi_status'] = 'Нейтральный'
        df.loc[df['rsi'] >= self.overbought, 'rsi_status'] = 'Перекупленность'
        df.loc[df['rsi'] <= self.oversold, 'rsi_status'] = 'Перепроданность'
        
        # Тренд MFI
        df['mfi_trend'] = 'Нейтральный'
        df.loc[df['bullish_cloud'], 'mfi_trend'] = 'Бычий'
        df.loc[df['bearish_cloud'], 'mfi_trend'] = 'Медвежий'
        
        # Моментум
        df['momentum'] = 'Падает'
        df.loc[df['rsi_change_1'] > 0, 'momentum'] = 'Растет'
        
        return df
    
    def generate_webhook_message(self, action: str, symbol: str, 
                                 price: float) -> Dict[str, Any]:
        """
        Генерация JSON сообщения для webhook (для бота)
        """
        return {
            "action": action,
            "symbol": symbol,
            "riskPercent": self.bot_risk,
            "leverage": self.bot_leverage,
            "takeProfitPercent": self.bot_tp,
            "stopLossPercent": self.bot_sl,
            "price": round(price, 2)
        }
    
    def get_latest_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Получение последних сигналов и состояния индикаторов
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'rsi_value': round(latest['rsi'], 2),
            'mfi_trend': latest['mfi_trend'],
            'rsi_status': latest['rsi_status'],
            'last_signal': latest['last_signal'],
            'momentum': latest['momentum'],
            'buy_signal': bool(latest['buy_signal']),
            'sell_signal': bool(latest['sell_signal']),
            'strong_bullish_signal': bool(latest['strong_bullish_signal']),
            'strong_bearish_signal': bool(latest['strong_bearish_signal']),
            'bullish_cloud': bool(latest['bullish_cloud']),
            'bearish_cloud': bool(latest['bearish_cloud']),
            'price': latest['close']
        }
    
    def process_data(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[pd.DataFrame, list]:
        """
        Полная обработка данных: расчет индикаторов, генерация сигналов и алертов
        
        Возвращает:
        - DataFrame с индикаторами и сигналами
        - Список алертов (webhook сообщения)
        """
        # Расчет индикаторов
        df = self.calculate_indicators(df)
        
        # Генерация сигналов
        df = self.generate_signals(df)
        
        # Генерация алертов
        alerts = []
        
        if self.enable_alerts and len(df) > 0:
            latest = df.iloc[-1]
            
            # Сильный бычий сигнал
            if latest['strong_bullish_signal'] and not self.last_strong_buy_signal:
                self.last_strong_buy_signal = True
                webhook_msg = self.generate_webhook_message(
                    "Buy", symbol, latest['close']
                )
                alerts.append({
                    'type': 'strong_buy',
                    'message': webhook_msg,
                    'text': f"🟢 STRONG BUY Signal | {symbol} | Price: {latest['close']:.2f}"
                })
            
            # Сильный медвежий сигнал
            if latest['strong_bearish_signal'] and not self.last_strong_sell_signal:
                self.last_strong_sell_signal = True
                webhook_msg = self.generate_webhook_message(
                    "Sell", symbol, latest['close']
                )
                alerts.append({
                    'type': 'strong_sell',
                    'message': webhook_msg,
                    'text': f"🔴 STRONG SELL Signal | {symbol} | Price: {latest['close']:.2f}"
                })
            
            # Обычный сигнал покупки
            if latest['buy_signal'] and not self.last_regular_buy_signal:
                self.last_regular_buy_signal = True
                alerts.append({
                    'type': 'buy',
                    'text': f"🟢 BUY Signal | {symbol} | Price: {latest['close']:.2f}"
                })
            
            # Обычный сигнал продажи
            if latest['sell_signal'] and not self.last_regular_sell_signal:
                self.last_regular_sell_signal = True
                alerts.append({
                    'type': 'sell',
                    'text': f"🔴 SELL Signal | {symbol} | Price: {latest['close']:.2f}"
                })
            
            # Сброс флагов при отсутствии условий
            if not latest['strong_bullish_signal']:
                self.last_strong_buy_signal = False
            if not latest['strong_bearish_signal']:
                self.last_strong_sell_signal = False
            if not latest['buy_signal']:
                self.last_regular_buy_signal = False
            if not latest['sell_signal']:
                self.last_regular_sell_signal = False
        
        return df, alerts


def example_usage():
    """
    Пример использования индикатора
    """
    # Создание тестовых данных
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    # Генерация случайных OHLCV данных
    close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(200) * 0.1,
        'high': close_prices + abs(np.random.randn(200) * 0.3),
        'low': close_prices - abs(np.random.randn(200) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Создание индикатора
    indicator = RSIMFIIndicator(
        rsi_length=14,
        oversold=30,
        overbought=70,
        require_volume=False,
        trend_confirmation=False
    )
    
    # Обработка данных
    result_df, alerts = indicator.process_data(df, symbol="BTCUSDT")
    
    # Получение последних сигналов
    latest_signals = indicator.get_latest_signals(result_df)
    
    print("=" * 60)
    print("ПОСЛЕДНИЕ СИГНАЛЫ")
    print("=" * 60)
    for key, value in latest_signals.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("АЛЕРТЫ")
    print("=" * 60)
    for alert in alerts:
        print(f"Тип: {alert['type']}")
        print(f"Сообщение: {alert['text']}")
        if 'message' in alert:
            print(f"Webhook: {alert['message']}")
        print("-" * 60)
    
    # Показать последние 5 строк с сигналами
    print("\n" + "=" * 60)
    print("ПОСЛЕДНИЕ ЗНАЧЕНИЯ")
    print("=" * 60)
    columns_to_show = ['close', 'rsi', 'mfi', 'buy_signal', 'sell_signal', 
                       'strong_bullish_signal', 'strong_bearish_signal']
    print(result_df[columns_to_show].tail())
    
    return result_df, indicator


if __name__ == "__main__":
    result_df, indicator = example_usage()
