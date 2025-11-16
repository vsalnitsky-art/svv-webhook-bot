from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests

def keep_alive():
    """Функция для поддержания сервера активным"""
    while True:
        try:
            # Пингуем себя каждые 10 минут
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
            print("🔄 Keep-alive ping sent")
        except:
            print("⚠️ Keep-alive ping failed")
        
        # Ждем 10 минут
        time.sleep(600)

# 🔐 БЕЗОПАСНЫЙ ИМПОРТ КЛЮЧЕЙ
from config import get_api_credentials, DEFAULT_LEVERAGE, DEFAULT_RISK_PERCENT

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔄 ЗАПУСКАЕМ KEEP-ALIVE ПРИ СТАРТЕ ПРИЛОЖЕНИЯ (вместо before_first_request)
keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.daemon = True
keep_alive_thread.start()

class BybitTradingBot:
    def __init__(self):
        try:
            # 🔐 БЕЗОПАСНОЕ ПОЛУЧЕНИЕ КЛЮЧЕЙ
            api_key, api_secret = get_api_credentials()
            
            self.session = HTTP(
                testnet=False,
                api_key=api_key,
                api_secret=api_secret,
            )
            logger.info("✅ Бот инициализирован с зашифрованными ключами")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации бота: {e}")
            raise

    def calculate_position_size(self, balance, risk_percent, leverage, price):
        """Расчет размера позиции в USDT и количества монет"""
        position_size_usdt = balance * (risk_percent / 100) * leverage
        quantity = position_size_usdt / price
        return round(quantity, 6), round(position_size_usdt, 2)

    def get_current_price(self, symbol):
        """Получение текущей цены с улучшенной обработкой ошибок"""
        try:
            logger.info(f"🔍 Получение цены для {symbol}")
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            
            # Детальная проверка структуры ответа
            logger.info(f"📊 Ответ API: {ticker}")
            
            if not ticker:
                logger.error("❌ Пустой ответ от API")
                return None
                
            if 'result' not in ticker:
                logger.error(f"❌ Нет 'result' в ответе: {ticker}")
                return None
                
            if 'list' not in ticker['result']:
                logger.error(f"❌ Нет 'list' в result: {ticker['result']}")
                return None
                
            if not ticker['result']['list']:
                logger.error(f"❌ Пустой список в list: {ticker['result']['list']}")
                return None
                
            first_item = ticker['result']['list'][0]
            if 'lastPrice' not in first_item:
                logger.error(f"❌ Нет 'lastPrice' в элементе: {first_item}")
                return None
                
            last_price_str = first_item['lastPrice']
            logger.info(f"🔢 Получена строка цены: '{last_price_str}'")
            
            # Проверяем, что цена не пустая и может быть конвертирована
            if last_price_str and last_price_str.strip() and last_price_str != 'None':
                # Убираем возможные пробелы и конвертируем
                price_value = last_price_str.strip()
                # Заменяем запятые на точки если есть
                price_value = price_value.replace(',', '.')
                try:
                    price = float(price_value)
                    logger.info(f"💰 Цена {symbol}: ${price}")
                    return price
                except ValueError as e:
                    logger.error(f"❌ Ошибка конвертации '{price_value}' в float: {e}")
                    return None
            else:
                logger.error(f"❌ Пустая или невалидная цена: '{last_price_str}'")
                return None
                
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка получения цены для {symbol}: {e}")
            return None

    def set_leverage(self, symbol, leverage):
        """Установка плеча"""
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            return True
        except Exception as e:
            logger.warning(f"Leverage setting warning: {e}")
            return False

    def normalize_symbol(self, symbol):
        """Нормализация символа из TradingView"""
        # Удаляем TradingView постфиксы
        symbol = symbol.replace('.P', '').replace('.PERP', '').replace('.D', '')
        
        # Берем только часть до точки (если есть)
        symbol = symbol.split('.')[0]
        
        # 🔥 ИСПРАВЛЯЕМ ОПЕЧАТКУ В СИМВОЛЕ
        symbol = symbol.replace('ADAMSDT', 'ADAUSDT')
        
        # Проверяем формат и добавляем USDT если нужно
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        # Приводим к верхнему регистру
        symbol = symbol.upper()
        
        return symbol

    def place_order(self, data):
        try:
            # Параметры из TradingView
            action = data.get('action', 'Buy')
            raw_symbol = data.get('symbol', 'BTCUSDT')
            
            # 🔄 НОРМАЛИЗАЦИЯ СИМВОЛА
            symbol = self.normalize_symbol(raw_symbol)
            
            logger.info(f"📈 Символ: {raw_symbol} -> {symbol}")

            leverage = data.get('leverage', 5)
            risk_percent = data.get('riskPercent', 1)
            tp_percent = data.get('takeProfitPercent', 3)
            sl_percent = data.get('stopLossPercent', 1.5)

            # Получаем реальный баланс
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            real_balance = float(balance_info['result']['list'][0]['totalAvailableBalance'])
            
            logger.info(f"💰 Реальный баланс: ${real_balance}")

            # Получение текущей цены
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": f"Не удалось получить цену для {symbol}"}

            logger.info(f"💰 Текущая цена: ${current_price}")

            # Установка плеча
            self.set_leverage(symbol, leverage)

            # 🔄 РАСЧЕТ ОТ РЕАЛЬНОГО БАЛАНСА
            quantity, position_size = self.calculate_position_size(
                real_balance,  # Используем реальный баланс!
                risk_percent, 
                leverage, 
                current_price
            )

            logger.info(f"📊 Рассчитано: Количество: {quantity}, Размер позиции: ${position_size}")

            # ФОРМАТИРОВАНИЕ КОЛИЧЕСТВА под разные пары
            if symbol in ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']:
                formatted_quantity = format(quantity, '.6f')
            elif symbol in ['BTCUSDT', 'ETHUSDT']:
                formatted_quantity = format(quantity, '.5f')  
            else:
                formatted_quantity = format(quantity, '.4f')

            logger.info(f"🔢 Отформатированное количество: {formatted_quantity}")

            # Проверка минимального объема
            if quantity < 0.0001:
                return {"status": "error", "error": f"Слишком маленький объем: {quantity}"}

            # Расчет TP/SL цен
            if action == "Buy":
                tp_price = round(current_price * (1 + tp_percent / 100), 2)
                sl_price = round(current_price * (1 - sl_percent / 100), 2)
                position_index = 0
            else:  # Sell
                tp_price = round(current_price * (1 - tp_percent / 100), 2)
                sl_price = round(current_price * (1 + sl_percent / 100), 2)
                position_index = 1

            logger.info(f"🎯 TP: ${tp_price}, SL: ${sl_price}")

            # Размещение рыночного ордера
            order = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=action,
                orderType="Market",
                qty=formatted_quantity,
                timeInForce="GTC",
            )

            order_id = order['result']['orderId']
            logger.info(f"✅ Ордер размещен: {order_id}")

            # Установка TP/SL
            if tp_percent > 0 or sl_percent > 0:
                try:
                    self.session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        takeProfit=str(tp_price),
                        stopLoss=str(sl_price),
                        positionIdx=position_index
                    )
                    logger.info(f"🛡️ TP/SL установлены: TP=${tp_price}, SL=${sl_price}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось установить TP/SL: {e}")

            return {
                "status": "success",
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": float(formatted_quantity),
                "entry_price": current_price,
                "position_size_usdt": position_size,
                "take_profit_price": tp_price,
                "stop_loss_price": sl_price,
                "leverage": leverage,
                "real_balance_used": real_balance,  # Показываем какой баланс использовался
                "risk_percent": risk_percent
            }

        except Exception as e:
            logger.error(f"❌ Ошибка ордера: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

@app.route('/')
def home():
    return "Trading Bot is Running! ✅ Use /webhook for TradingView"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Bot is running"})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        logger.info(f"📨 Webhook вызван: {data}")
        result = bot.place_order(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"💥 Ошибка webhook: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
