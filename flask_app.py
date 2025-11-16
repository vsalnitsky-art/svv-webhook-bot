from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import os
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BybitTradingBot:
    def __init__(self):
        try:
            self.session = HTTP(
                testnet=False,
                api_key=os.environ.get('BYBIT_API_KEY'),
                api_secret=os.environ.get('BYBIT_API_SECRET'),
            )
            logger.info("✅ Бот инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")

    def place_order(self, data):
        try:
            logger.info(f"📨 Получен запрос: {data}")
            
            # Простая проверка связи
            ticker = self.session.get_tickers(category="linear", symbol="BTCUSDT")
            price = ticker['result']['list'][0]['lastPrice']
            
            return {
                "status": "success",
                "message": "Тест связи успешен!",
                "btc_price": price,
                "received_data": data
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка в place_order: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

@app.route('/')
def home():
    return "Trading Bot is Running! ✅"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Bot is running"})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        logger.info(f"🔧 Webhook вызван с данными: {data}")
        result = bot.place_order(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"💥 Ошибка webhook: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
