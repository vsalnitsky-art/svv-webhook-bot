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
        try:
            action = data.get('action', 'BUY').upper()
            symbol = data.get('symbol', 'BTCUSDT')
            leverage = data.get('leverage', 10)
            risk_percent = data.get('riskPercent', 2)
            account_balance = data.get('accountBalance', 1000)
            tp_percent = data.get('takeProfitPercent', 5)
            sl_percent = data.get('stopLossPercent', 2)
            
            logger.info(f"Processing: {action} {symbol} {leverage}x {risk_percent}%")
            
            # Установка плеча
            self.session.set_leverage(
                category="linear", 
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            
            # Получение цены и расчет объема
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": "Could not get price"}
            
            quantity, position_size = self.calculate_position_size(
                account_balance, risk_percent, leverage, current_price
            )
            
            # Расчет TP/SL
            if action == "BUY":
                tp_price = round(current_price * (1 + tp_percent / 100), 2)
                sl_price = round(current_price * (1 - sl_percent / 100), 2)
            else:
                tp_price = round(current_price * (1 - tp_percent / 100), 2)
                sl_price = round(current_price * (1 + sl_percent / 100), 2)
            
            # Размещение ордера
            order = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=action,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC",
            )
            
            # Установка TP/SL
            if tp_percent > 0 or sl_percent > 0:
                position_index = 0 if action == "BUY" else 1
                self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=str(tp_price),
                    stopLoss=str(sl_price),
                    positionIdx=position_index
                )
            
            return {
                "status": "success",
                "order_id": order['result']['orderId'],
                "symbol": symbol,
                "quantity": quantity,
                "position_size_usdt": position_size,
                "take_profit": tp_price,
                "stop_loss": sl_price
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
