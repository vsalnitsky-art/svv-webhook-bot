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

# 🔄 ЗАПУСКАЕМ KEEP-ALIVE ПРИ СТАРТЕ ПРИЛОЖЕНИЯ
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
        """Получение текущей цены с несколькими fallback методами"""
        methods = [
            self._get_price_from_tickers,
            self._get_price_from_orderbook,
            self._get_price_from_public_trading
        ]
        
        for method in methods:
            price = method(symbol)
            if price is not None:
                logger.info(f"✅ Цена получена методом {method.__name__}: ${price}")
                return price
        
        logger.error(f"❌ Все методы получения цены для {symbol} не сработали")
        return None

    def _get_price_from_tickers(self, symbol):
        """Попытка получить цену через get_tickers"""
        try:
            logger.info(f"🔍 Попытка получить цену через get_tickers для {symbol}")
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            
            # Детальная диагностика ответа
            logger.info(f"📊 Полный ответ get_tickers: {ticker}")
            
            if not ticker or 'result' not in ticker:
                logger.error("❌ Нет результата в ответе get_tickers")
                return None
                
            result = ticker['result']
            if 'list' not in result or not result['list']:
                logger.error("❌ Пустой список в ответе get_tickers")
                return None
                
            first_item = result['list'][0]
            logger.info(f"📋 Первый элемент списка: {first_item}")
            
            # Пробуем разные возможные поля с ценой
            price_fields = ['lastPrice', 'markPrice', 'indexPrice', 'prevPrice24h']
            
            for field in price_fields:
                if field in first_item:
                    price_str = first_item[field]
                    logger.info(f"🔍 Найдено поле {field}: '{price_str}'")
                    
                    if price_str and price_str.strip() and price_str not in ['', 'None', 'null']:
                        try:
                            # Очистка строки от лишних символов
                            clean_price = price_str.strip().replace(',', '')
                            price = float(clean_price)
                            logger.info(f"💰 Цена из {field}: ${price}")
                            return price
                        except ValueError as e:
                            logger.warning(f"⚠️ Не удалось конвертировать {field} '{price_str}': {e}")
                            continue
            
            logger.error("❌ Ни одно поле цены не содержит валидных данных")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка в _get_price_from_tickers: {e}")
            return None

    def _get_price_from_orderbook(self, symbol):
        """Попытка получить цену через стакан цен"""
        try:
            logger.info(f"🔍 Попытка получить цену через стакан для {symbol}")
            orderbook = self.session.get_orderbook(category="linear", symbol=symbol)
            
            if (orderbook and 'result' in orderbook and 
                'a' in orderbook['result'] and orderbook['result']['a']):
                # Берем лучшую цену продажи (ask price)
                ask_price_str = orderbook['result']['a'][0][0]
                if ask_price_str and ask_price_str.strip():
                    clean_price = ask_price_str.strip().replace(',', '')
                    price = float(clean_price)
                    logger.info(f"💰 Цена из стакана: ${price}")
                    return price
                    
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить цену из стакана: {e}")
            
        return None

    def _get_price_from_public_trading(self, symbol):
        """Попытка получить цену через публичные сделки"""
        try:
            logger.info(f"🔍 Попытка получить цену через публичные сделки для {symbol}")
            trades = self.session.get_public_trade_history(
                category="linear", 
                symbol=symbol, 
                limit=1
            )
            
            if (trades and 'result' in trades and 
                'list' in trades['result'] and trades['result']['list']):
                last_trade = trades['result']['list'][0]
                price_str = last_trade.get('price')
                if price_str and price_str.strip():
                    clean_price = price_str.strip().replace(',', '')
                    price = float(clean_price)
                    logger.info(f"💰 Цена из последней сделки: ${price}")
                    return price
                    
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить цену из сделок: {e}")
            
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
                real_balance,
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
                tp_price = round(current_price * (1 + tp_percent / 100), 4)
                sl_price = round(current_price * (1 - sl_percent / 100), 4)
                position_index = 0
            else:  # Sell
                tp_price = round(current_price * (1 - tp_percent / 100), 4)
                sl_price = round(current_price * (1 + sl_percent / 100), 4)
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
                "real_balance_used": real_balance,
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
