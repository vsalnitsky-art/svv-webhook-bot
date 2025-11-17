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

from config import get_api_credentials, DEFAULT_LEVERAGE, DEFAULT_RISK_PERCENT

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

    def calculate_position_size(self, balance, risk_percent, leverage, price):
        """Расчет размера позиции в USDT и количества монет"""
        position_size_usdt = balance * (risk_percent / 100) * leverage
        quantity = position_size_usdt / price
        return round(quantity, 6), round(position_size_usdt, 2)

    def validate_symbol(self, symbol):
        """Проверка что символ существует и доступен для торговли"""
        try:
            # Пробуем получить информацию о символе
            info = self.session.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if (info and info.get('retCode') == 0 and 
                info.get('result') and 
                info['result'].get('list') and 
                len(info['result']['list']) > 0):
                
                symbol_info = info['result']['list'][0]
                status = symbol_info.get('status', '')
                
                if status == 'Trading':
                    logger.info(f"✅ Символ {symbol} доступен для торговли")
                    return True
                else:
                    logger.error(f"❌ Символ {symbol} не доступен для торговли. Статус: {status}")
                    return False
            else:
                logger.error(f"❌ Символ {symbol} не найден")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка проверки символа {symbol}: {e}")
            return False

    def get_current_price(self, symbol):
        """Улучшенное получение цены с проверкой символа"""
        # Сначала проверяем что символ существует
        if not self.validate_symbol(symbol):
            return None

        methods = [
            self._get_price_via_tickers,
            self._get_price_via_orderbook, 
            self._get_price_via_kline,
            self._get_price_via_public_trades
        ]
        
        for i, method in enumerate(methods):
            try:
                logger.info(f"🔍 Метод {i+1}: {method.__name__} для {symbol}")
                price = method(symbol)
                if price and price > 0:
                    logger.info(f"✅ Цена получена: ${price}")
                    return price
            except Exception as e:
                logger.warning(f"⚠️ Метод {i+1} не сработал: {e}")
                continue
        
        logger.error(f"❌ Все методы получения цены для {symbol} провалились")
        return None

    def _get_price_via_tickers(self, symbol):
        """Основной метод через get_tickers"""
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
            logger.info(f"📊 Ответ get_tickers: {response}")
            
            if response.get('retCode') != 0:
                logger.error(f"❌ Ошибка API: {response.get('retMsg')}")
                return None
                
            result = response.get('result', {})
            tickers_list = result.get('list', [])
            
            if not tickers_list:
                logger.error("❌ Пустой список тикеров")
                return None
                
            ticker = tickers_list[0]
            logger.info(f"📋 Данные тикера: {ticker}")
            
            # Пробуем все возможные поля с ценой
            price_fields = ['lastPrice', 'markPrice', 'indexPrice', 'prevPrice24h', 'highPrice24h', 'lowPrice24h']
            
            for field in price_fields:
                price_str = ticker.get(field)
                if price_str:
                    logger.info(f"🔍 Поле {field}: '{price_str}'")
                    try:
                        clean_price = str(price_str).strip().replace(',', '').replace(' ', '')
                        if clean_price and clean_price not in ['', 'None', 'null', '0']:
                            price = float(clean_price)
                            if price > 0:
                                return price
                    except (ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Ошибка конвертации {field}: {e}")
                        continue
            
            logger.error("❌ Ни одно поле цены не содержит валидных данных")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка в _get_price_via_tickers: {e}")
            return None

    def _get_price_via_orderbook(self, symbol):
        """Через стакан цен"""
        try:
            response = self.session.get_orderbook(category="linear", symbol=symbol, limit=1)
            
            if response.get('retCode') == 0:
                result = response.get('result', {})
                # Пробуем цену продажи (ask)
                asks = result.get('a', [])
                if asks and asks[0]:
                    price_str = asks[0][0]
                    if price_str:
                        return float(price_str)
                        
                # Или цену покупки (bid)
                bids = result.get('b', [])
                if bids and bids[0]:
                    price_str = bids[0][0]
                    if price_str:
                        return float(price_str)
                        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка в _get_price_via_orderbook: {e}")
            
        return None

    def _get_price_via_kline(self, symbol):
        """Через последний свечной график"""
        try:
            from datetime import datetime, timedelta
            
            # Получаем последнюю завершенную свечу
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000)
            
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=1,  # 1 минута
                start=start_time,
                end=end_time,
                limit=1
            )
            
            if response.get('retCode') == 0:
                result = response.get('result', {})
                klines = result.get('list', [])
                if klines and klines[0]:
                    # Берем цену закрытия последней свечи
                    close_price = klines[0][4]  # [open, high, low, close, volume, ...]
                    if close_price:
                        return float(close_price)
                        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка в _get_price_via_kline: {e}")
            
        return None

    def _get_price_via_public_trades(self, symbol):
        """Через последние публичные сделки"""
        try:
            response = self.session.get_public_trade_history(
                category="linear",
                symbol=symbol,
                limit=5  # Берем несколько последних сделок
            )
            
            if response.get('retCode') == 0:
                result = response.get('result', {})
                trades = result.get('list', [])
                if trades:
                    # Берем цену последней сделки
                    last_trade = trades[0]
                    price_str = last_trade.get('price')
                    if price_str:
                        return float(price_str)
                        
        except Exception as e:
            logger.warning(f"⚠️ Ошибка в _get_price_via_public_trades: {e}")
            
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
        """Нормализация символа"""
        symbol = symbol.replace('.P', '').replace('.PERP', '').replace('.D', '')
        symbol = symbol.split('.')[0]
        symbol = symbol.replace('ADAMSDT', 'ADAUSDT')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        symbol = symbol.upper()
        logger.info(f"🔧 Нормализованный символ: {symbol}")
        return symbol

    def place_order(self, data):
        try:
            action = data.get('action', 'Buy')
            raw_symbol = data.get('symbol', 'BTCUSDT')
            
            symbol = self.normalize_symbol(raw_symbol)
            logger.info(f"📈 Символ: {raw_symbol} -> {symbol}")

            leverage = min(data.get('leverage', 5), 25)
            risk_percent = min(data.get('riskPercent', 1), 10)
            tp_percent = data.get('takeProfitPercent', 3)
            sl_percent = data.get('stopLossPercent', 1.5)

            # Получаем баланс
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            real_balance = float(balance_info['result']['list'][0]['totalAvailableBalance'])
            logger.info(f"💰 Реальный баланс: ${real_balance}")

            # Получение цены
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": f"Не удалось получить цену для {symbol}. Символ может не существовать."}

            logger.info(f"💰 Текущая цена: ${current_price}")

            # Расчет позиции
            quantity, position_size = self.calculate_position_size(
                real_balance, risk_percent, leverage, current_price
            )

            if quantity < 0.0001:
                return {"status": "error", "error": f"Слишком маленький объем: {quantity}"}

            # Форматирование количества
            if symbol in ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']:
                formatted_quantity = format(quantity, '.6f')
            elif symbol in ['BTCUSDT', 'ETHUSDT']:
                formatted_quantity = format(quantity, '.5f')  
            else:
                formatted_quantity = format(quantity, '.4f')

            # Расчет TP/SL
            if action == "Buy":
                tp_price = round(current_price * (1 + tp_percent / 100), 4)
                sl_price = round(current_price * (1 - sl_percent / 100), 4)
                position_index = 0
            else:
                tp_price = round(current_price * (1 - tp_percent / 100), 4)
                sl_price = round(current_price * (1 + sl_percent / 100), 4)
                position_index = 1

            # Размещение ордера
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
    app.run(host='0.0.0.0', port=10000, debug=False)
