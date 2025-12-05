import threading
import time
import logging
import pandas as pd
import pandas_ta as ta
from settings_manager import settings
from strategy_ob_trend import ob_trend_strategy

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.data = {} # Кеш даних для UI: {symbol: {'rsi': 55, 'exit_status': 'Safe', ...}}
        self.running = True
        
        # Запускаємо моніторинг у фоновому потоці
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🛡️ Active Position Monitor & Smart Exit Started")

    def loop(self):
        """Головний цикл перевірки активних угод"""
        while self.running:
            try:
                # 1. Отримуємо всі позиції з біржі (USDT Perpetual)
                resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
                
                if resp['retCode'] == 0:
                    all_positions = resp['result']['list']
                    # Фільтруємо тільки активні угоди (розмір > 0)
                    active_positions = [p for p in all_positions if float(p['size']) > 0]
                    
                    # Очищаємо кеш від закритих угод (Garbage Collection)
                    active_symbols = [p['symbol'] for p in active_positions]
                    keys_to_remove = [k for k in self.data.keys() if k not in active_symbols]
                    for k in keys_to_remove:
                        del self.data[k]

                    # 2. Обробляємо кожну активну позицію
                    for pos in active_positions:
                        self._process_position(pos)
                        
                    # Пауза між циклами оновлення (береться з конфігу, зазвичай 5-10 сек)
                    time.sleep(self.config.get('SCANNER_INTERVAL', 10))
                else:
                    logger.warning(f"⚠️ Error getting positions: {resp['retMsg']}")
                    time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Scanner Loop Error: {e}")
                time.sleep(10)

    def _process_position(self, pos):
        """
        Аналіз однієї позиції:
        1. Завантаження свічок HTF.
        2. Перевірка RSI та умов виходу.
        3. Виконання закриття, якщо потрібно.
        """
        symbol = pos['symbol']
        side = pos['side'] # "Buy" (Long) або "Sell" (Short)
        
        # Ініціалізація структури даних, якщо це нова угода
        if symbol not in self.data:
            self.data[symbol] = {
                'rsi': 0, 
                'exit_status': 'Safe', 
                'exit_details': '-', 
                'pressure': 0
            }

        try:
            # А. Визначаємо HTF (Глобальний таймфрейм з налаштувань)
            htf = str(settings.get("htfSelection", "240")) # Default 4H
            
            # Б. Завантажуємо свічки для цього ТФ
            df = self.fetch_candles(symbol, htf)

            if df is not None:
                # В. Передаємо дані в Стратегію для перевірки на вихід
                # check_exit_signal повертає: {'close': True/False, 'reason': '...', 'details': {...}}
                exit_info = ob_trend_strategy.check_exit_signal(df, side)
                
                # Г. Оновлюємо дані для UI
                rsi_val = exit_info.get('details', {}).get('rsi', 0)
                self.data[symbol]['rsi'] = rsi_val
                
                # Д. Логіка Статусу та Виходу
                if exit_info['close']:
                    self.data[symbol]['exit_status'] = 'EXIT NOW'
                    self.data[symbol]['exit_details'] = exit_info['reason']
                    
                    # Е. АВТОМАТИЧНЕ ЗАКРИТТЯ (Якщо увімкнено "Smart Exit")
                    if settings.get("exit_enableStrategy", False):
                        logger.info(f"🚨 SMART EXIT TRIGGERED: {symbol} ({side}) -> {exit_info['reason']}")
                        
                        # Відправляємо ордер на закриття (Close)
                        res = self.bot.place_order({
                            "action": "Close",
                            "symbol": symbol,
                            "direction": "Long" if side == "Buy" else "Short"
                        })
                        
                        if res.get('status') == 'ok':
                            logger.info(f"✅ Position {symbol} Closed Successfully via Smart Exit")
                        else:
                            logger.error(f"❌ Failed to close {symbol}: {res.get('reason')}")
                else:
                    # Логіка Попереджень (Warning Zone)
                    # Якщо RSI підходить близько до лімітів, показуємо Warning
                    limit_buy = float(settings.get('exit_rsiOverbought', 70))
                    limit_sell = float(settings.get('exit_rsiOversold', 30))
                    
                    status = "Safe"
                    # Для Long: якщо RSI >= 65 (при ліміті 70) -> Warning
                    if side == "Buy" and rsi_val >= (limit_buy - 5): status = "Warning"
                    # Для Short: якщо RSI <= 35 (при ліміті 30) -> Warning
                    if side == "Sell" and rsi_val <= (limit_sell + 5): status = "Warning"
                    
                    self.data[symbol]['exit_status'] = status
                    self.data[symbol]['exit_details'] = f"RSI: {rsi_val}"

        except Exception as e:
            # Не спамимо в лог, якщо помилка одинична
            # logger.error(f"Error processing {symbol}: {e}")
            pass

    def fetch_candles(self, symbol, timeframe, limit=50):
        """Завантажує свічки з Bybit з правильним мапінгом таймфреймів"""
        try:
            # Мапінг: Налаштування -> Bybit API
            tf_map = {
                '5': '5', '15': '15', '30': '30', 
                '45': '15', # 45m немає в API, беремо 15m для приблизної оцінки
                '60': '60', '240': '240', 'D': 'D'
            }
            req_tf = tf_map.get(str(timeframe), '240')
            
            response = self.bot.session.get_kline(
                category="linear", 
                symbol=symbol, 
                interval=req_tf, 
                limit=limit
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                # Bybit повертає: [time, open, high, low, close, volume, turnover]
                # Сортування: від нових до старих
                raw_data = response['result']['list']
                
                df = pd.DataFrame(raw_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                
                # Конвертація
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Реверс: старі -> нові (потрібно для pandas_ta)
                return df.iloc[::-1].reset_index(drop=True)
                
        except Exception:
            pass
        return None

    # === Getters для UI (використовуються в main_app.py) ===
    def get_coin_data(self, symbol):
        """Повертає весь об'єкт даних для монети"""
        return self.data.get(symbol, {})

    def get_current_rsi(self, symbol):
        """Повертає тільки RSI"""
        return self.data.get(symbol, {}).get('rsi', 0)

    def get_market_pressure(self, symbol):
        """Заглушка для тиску (Pressure), якщо знадобиться пізніше"""
        return self.data.get(symbol, {}).get('pressure', 0)
    
    def get_exit_status(self, symbol):
        """Повертає текстовий статус: Safe, Warning, EXIT NOW"""
        return self.data.get(symbol, {}).get('exit_status', 'N/A')