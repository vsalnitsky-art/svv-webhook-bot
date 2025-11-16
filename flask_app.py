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
        # Ключи из переменных окружения
        api_key = os.environ.get('BYBIT_API_KEY')
        api_secret = os.environ.get('BYBIT_API_SECRET')
        
        self.session = HTTP(
            testnet=False,  # Mainnet
            api_key=api_key,
            api_secret=api_secret,
        )
    
    def calculate_position_size(self, balance, risk_percent, leverage, price):
        position_size_usdt = balance * (risk_percent / 100) * leverage
        quantity = position_size_usdt / price
        return round(quantity, 6), round(position_size_usdt, 2)
    
    def get_current_price(self, symbol):
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return float(ticker['result']['list'][0]['lastPrice'])
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
def place_order(self, data):
    """Размещение ордера с TP/SL в процентах"""
    try:
        # Параметры из TradingView
        action = data.get('action', 'Buy').capitalize()  # Fix: "Buy" вместо "BUY"
        symbol = data.get('symbol', 'BTCUSDT')
        leverage = data.get('leverage', 10)
        risk_percent = data.get('riskPercent', 2)
        account_balance = data.get('accountBalance', 1000)
        tp_percent = data.get('takeProfitPercent', 5)
        sl_percent = data.get('stopLossPercent', 2)
        
        # Проверка действия
        if action not in ['Buy', 'Sell']:
            return {"status": "error", "error": f"Invalid action: {action}. Use 'Buy' or 'Sell'"}
        
        logger.info(f"Processing order: {action} {symbol}")
        logger.info(f"Params: Leverage: {leverage}x, Risk: {risk_percent}%, Balance: ${account_balance}")
        
        # Получение текущей цены
        current_price = self.get_current_price(symbol)
        if not current_price:
            return {"status": "error", "error": "Could not get current price"}
        
        logger.info(f"Current price: ${current_price}")
        
        # Установка плеча
        self.set_leverage(symbol, leverage)
        
        # Расчет объема
        quantity, position_size = self.calculate_position_size(
            account_balance, risk_percent, leverage, current_price
        )
        
        logger.info(f"Calculated: Quantity: {quantity}, Position size: ${position_size}")
        
        # Проверка минимального объема
        if quantity < 0.001:
            return {"status": "error", "error": f"Quantity too small: {quantity}"}
        
        # Расчет TP/SL цен в процентах
        if action == "Buy":
            tp_price = round(current_price * (1 + tp_percent / 100), 2)
            sl_price = round(current_price * (1 - sl_percent / 100), 2)
        else:  # Sell
            tp_price = round(current_price * (1 - tp_percent / 100), 2)
            sl_price = round(current_price * (1 + sl_percent / 100), 2)
        
        logger.info(f"TP price: ${tp_price}, SL price: ${sl_price}")
        
        # Размещение рыночного ордера
        order = self.session.place_order(
            category="linear",
            symbol=symbol,
            side=action,  # Теперь "Buy" или "Sell"
            orderType="Market",
            qty=str(quantity),
            timeInForce="GTC",
        )
        
        order_id = order['result']['orderId']
        logger.info(f"Order placed successfully: {order_id}")
        
        # Установка TP/SL
        if tp_percent > 0 or sl_percent > 0:
            try:
                position_index = 0 if action == "Buy" else 1
                self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=str(tp_price),
                    stopLoss=str(sl_price),
                    positionIdx=position_index
                )
                logger.info(f"TP/SL set successfully: TP=${tp_price}, SL=${sl_price}")
            except Exception as e:
                logger.warning(f"Could not set TP/SL: {e}")
        
        return {
            "status": "success",
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "entry_price": current_price,
            "position_size_usdt": position_size,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "leverage": leverage
        }
        
    except Exception as e:
        logger.error(f"Order error: {e}")
        return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

@app.route('/')
def home():
    return "Trading Bot is Running! Use /webhook for TradingView alerts."

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        result = bot.place_order(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Bot is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

@app.route('/test-connection', methods=['GET'])
def test_connection():
    """Тест подключения к Bybit"""
    try:
        # Простой тест без торговли
        ticker = bot.session.get_tickers(category="linear", symbol="BTCUSDT")
        btc_price = ticker['result']['list'][0]['lastPrice']
        
        return jsonify({
            "status": "success", 
            "message": "Connection to Bybit successful!",
            "btc_price": btc_price,
            "timestamp": "2024-11-16 14:30:00"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
