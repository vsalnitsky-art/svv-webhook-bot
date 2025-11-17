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
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
            print("🔄 Keep-alive ping sent")
        except:
            print("⚠️ Keep-alive ping failed")
        time.sleep(600)

from config import get_api_credentials

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.daemon = True
keep_alive_thread.start()

class BybitTradingBot:
    def __init__(self):
        try:
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

    def normalize_symbol(self, symbol):
        """Нормализация символа"""
        symbol = symbol.replace('.P', '').replace('.PERP', '').replace('.D', '')
        symbol = symbol.split('.')[0]
        symbol = symbol.replace('ADAMSDT', 'ADAUSDT')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        symbol = symbol.upper()
        logger.info(f"🔧 Нормализованный символ: {symbol}")
        return symbol

    def get_current_price(self, symbol):
        """Получение текущей цены"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            
            if (ticker and ticker.get('retCode') == 0 and 
                ticker.get('result') and 
                ticker['result'].get('list') and 
                len(ticker['result']['list']) > 0):
                
                last_price_str = ticker['result']['list'][0].get('lastPrice')
                
                if last_price_str and last_price_str.strip():
                    price = float(last_price_str.strip())
                    logger.info(f"💰 Цена {symbol}: ${price}")
                    return price
                    
            logger.error(f"❌ Не удалось получить цену для {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения цены: {e}")
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

    def place_simple_order(self, data):
        """Простое открытие сделки на фиксированную сумму 5 USDT"""
        try:
            # Параметры из TradingView
            action = data.get('action', 'Buy')
            raw_symbol = data.get('symbol', 'BTCUSDT')
            leverage = data.get('leverage', 5)
            
            # 🔄 НОРМАЛИЗАЦИЯ СИМВОЛА
            symbol = self.normalize_symbol(raw_symbol)
            
            logger.info(f"🎯 АЛЕРТ: {action} {symbol} с плечом {leverage}x")

            # Установка плеча
            self.set_leverage(symbol, leverage)

            # Получение текущей цены
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": f"Не удалось получить цену для {symbol}"}

            # 🔥 ФИКСИРОВАННАЯ СУММА 6 USDT
            fixed_amount_usdt = 6.0
            
            # Расчет количества монет
            quantity = fixed_amount_usdt / current_price
            
            # Форматирование количества
            if symbol in ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']:
                formatted_quantity = format(quantity, '.6f')
            elif symbol in ['BTCUSDT', 'ETHUSDT']:
                formatted_quantity = format(quantity, '.5f')  
            else:
                formatted_quantity = format(quantity, '.4f')

            logger.info(f"💰 Фиксированная сумма: ${fixed_amount_usdt}")
            logger.info(f"📊 Количество: {formatted_quantity}")

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

            return {
                "status": "success",
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": float(formatted_quantity),
                "entry_price": current_price,
                "position_size_usdt": fixed_amount_usdt,
                "leverage": leverage,
                "message": f"Сделка открыта на ${fixed_amount_usdt}"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка ордера: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

@app.route('/')
def home():
    return "Simple Trading Bot is Running! ✅ Use /webhook for TradingView alerts"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Bot is running"})

@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    try:
        # Логируем заголовки для диагностики
        logger.info(f"📨 Headers: {dict(request.headers)}")
        logger.info(f"📨 Content-Type: {request.content_type}")
        logger.info(f"📨 Method: {request.method}")
        
        # Пробуем разные методы получения данных
        data = None
        
        # Метод 1: JSON данные
        if request.is_json:
            data = request.get_json()
            logger.info(f"📨 Данные (JSON): {data}")
        
        # Метод 2: Form данные
        elif request.form:
            data = request.form.to_dict()
            logger.info(f"📨 Данные (FORM): {data}")
        
        # Метод 3: Raw данные (текст)
        elif request.data:
            try:
                raw_data = request.get_data(as_text=True)
                logger.info(f"📨 Raw данные: {raw_data}")
                # Пробуем распарсить как JSON
                import json
                data = json.loads(raw_data)
                logger.info(f"📨 Данные (RAW->JSON): {data}")
            except:
                # Если не JSON, пробуем парсить как query string
                try:
                    from urllib.parse import parse_qs
                    parsed = parse_qs(raw_data)
                    data = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
                    logger.info(f"📨 Данные (RAW->FORM): {data}")
                except:
                    data = {"raw": raw_data}
        
        # Метод 4: Query параметры (для GET запросов)
        elif request.args:
            data = request.args.to_dict()
            logger.info(f"📨 Данные (QUERY): {data}")
        
        if not data:
            return jsonify({"status": "error", "error": "No data received"}), 400
        
        logger.info(f"📨 Обработанные данные: {data}")
        result = bot.place_simple_order(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 Ошибка webhook: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
