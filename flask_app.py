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

    def get_available_balance(self):
        """Получение доступного баланса с обработкой пустых значений"""
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            logger.info(f"💰 Полный ответ баланса: {balance_info}")
            
            if balance_info.get('retCode') != 0:
                logger.error(f"❌ Ошибка API баланса: {balance_info.get('retMsg')}")
                return None
            
            result = balance_info.get('result', {})
            if not result or 'list' not in result or not result['list']:
                logger.error("❌ Пустой список в ответе баланса")
                return None
            
            account_list = result['list']
            logger.info(f"💰 Список аккаунтов: {account_list}")
            
            # Ищем USDT баланс
            for account in account_list:
                logger.info(f"💰 Обрабатываем аккаунт: {account}")
                
                # Пробуем разные поля с балансом
                balance_fields = [
                    'totalAvailableBalance',
                    'totalWalletBalance', 
                    'totalEquity',
                    'totalMarginBalance',
                    'totalPerpUPL'
                ]
                
                for field in balance_fields:
                    if field in account:
                        balance_str = account[field]
                        logger.info(f"💰 Поле {field}: '{balance_str}'")
                        
                        if balance_str and balance_str.strip() and balance_str not in ['', 'None', 'null']:
                            try:
                                clean_balance = str(balance_str).strip().replace(',', '').replace(' ', '')
                                balance = float(clean_balance)
                                if balance > 0:
                                    logger.info(f"💰 Найден баланс в поле {field}: ${balance}")
                                    return balance
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️ Ошибка конвертации {field}: {e}")
                                continue
                
                # Также проверяем монеты в аккаунте
                if 'coin' in account and account['coin']:
                    for coin in account['coin']:
                        logger.info(f"💰 Данные монеты: {coin}")
                        if coin.get('coin') == 'USDT':
                            coin_fields = [
                                'availableToWithdraw',
                                'walletBalance',
                                'equity',
                                'free'
                            ]
                            for field in coin_fields:
                                if field in coin:
                                    balance_str = coin[field]
                                    logger.info(f"💰 Монета USDT поле {field}: '{balance_str}'")
                                    
                                    if balance_str and balance_str.strip() and balance_str not in ['', 'None', 'null']:
                                        try:
                                            clean_balance = str(balance_str).strip().replace(',', '').replace(' ', '')
                                            balance = float(clean_balance)
                                            if balance > 0:
                                                logger.info(f"💰 Найден USDT баланс в поле {field}: ${balance}")
                                                return balance
                                        except (ValueError, TypeError) as e:
                                            logger.warning(f"⚠️ Ошибка конвертации USDT {field}: {e}")
                                            continue
            
            logger.error("❌ Не удалось найти доступный баланс USDT")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return None

    def validate_symbol(self, symbol):
        """Проверка что символ существует и доступен для торговли"""
        try:
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
            # 📝 ЛОГИРОВАНИЕ ВХОДНЫХ ДАННЫХ
            logger.info("📥" + "="*50)
            logger.info("📥 ВХОДНЫЕ ДАННЫЕ ОТ TRADINGVIEW:")
            logger.info(f"📥 Raw data: {data}")
            logger.info(f"📥 Action: {data.get('action')}")
            logger.info(f"📥 Symbol: {data.get('symbol')}")
            logger.info(f"📥 Leverage: {data.get('leverage')}")
            logger.info(f"📥 RiskPercent: {data.get('riskPercent')}")
            logger.info(f"📥 TakeProfitPercent: {data.get('takeProfitPercent')}")
            logger.info(f"📥 StopLossPercent: {data.get('stopLossPercent')}")
            logger.info("📥" + "="*50)

            action = data.get('action', 'Buy')
            raw_symbol = data.get('symbol', 'BTCUSDT')
            
            symbol = self.normalize_symbol(raw_symbol)
            logger.info(f"📈 Символ: {raw_symbol} -> {symbol}")

            leverage = min(data.get('leverage', 5), 25)
            risk_percent = min(data.get('riskPercent', 1), 10)
            tp_percent = data.get('takeProfitPercent', 3)
            sl_percent = data.get('stopLossPercent', 1.5)

            # Получаем баланс через улучшенный метод
            real_balance = self.get_available_balance()
            if real_balance is None:
                return {"status": "error", "error": "Не удалось получить доступный баланс"}
            
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

            # 📝 ЛОГИРОВАНИЕ РАСЧЕТОВ
            logger.info("🧮" + "="*50)
            logger.info("🧮 РАСЧЕТЫ ПОЗИЦИИ:")
            logger.info(f"🧮 Баланс: ${real_balance}")
            logger.info(f"🧮 Риск: {risk_percent}%")
            logger.info(f"🧮 Плечо: {leverage}x")
            logger.info(f"🧮 Цена: ${current_price}")
            logger.info(f"🧮 Размер позиции (USDT): ${position_size}")
            logger.info(f"🧮 Количество (монет): {quantity}")
            logger.info("🧮" + "="*50)

            if quantity < 0.0001:
                return {"status": "error", "error": f"Слишком маленький объем: {quantity}"}

            # Форматирование количества
            if symbol in ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']:
                formatted_quantity = format(quantity, '.6f')
            elif symbol in ['BTCUSDT', 'ETHUSDT']:
                formatted_quantity = format(quantity, '.5f')  
            else:
                formatted_quantity = format(quantity, '.4f')

            logger.info(f"🔢 Отформатированное количество: {formatted_quantity}")

            # Расчет TP/SL
            if action == "Buy":
                tp_price = round(current_price * (1 + tp_percent / 100), 4)
                sl_price = round(current_price * (1 - sl_percent / 100), 4)
                position_index = 0
            else:
                tp_price = round(current_price * (1 - tp_percent / 100), 4)
                sl_price = round(current_price * (1 + sl_percent / 100), 4)
                position_index = 1

            # 📝 ЛОГИРОВАНИЕ ТОРГОВЫХ ПАРАМЕТРОВ
            logger.info("🎯" + "="*50)
            logger.info("🎯 ТОРГОВЫЕ ПАРАМЕТРЫ:")
            logger.info(f"🎯 Действие: {action}")
            logger.info(f"🎯 Символ: {symbol}")
            logger.info(f"🎯 Плечо: {leverage}")
            logger.info(f"🎯 Текущая цена: ${current_price}")
            logger.info(f"🎯 Take Profit: ${tp_price} ({tp_percent}%)")
            logger.info(f"🎯 Stop Loss: ${sl_price} ({sl_percent}%)")
            logger.info(f"🎯 Количество: {formatted_quantity}")
            logger.info(f"🎯 Position Index: {position_index}")
            logger.info("🎯" + "="*50)

            # Размещение ордера
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": action,
                "orderType": "Market",
                "qty": formatted_quantity,
                "timeInForce": "GTC",
            }
            
            logger.info("📤" + "="*50)
            logger.info("📤 ДАННЫЕ ДЛЯ ОТПРАВКИ НА БИРЖУ:")
            logger.info(f"📤 Параметры ордера: {order_params}")
            logger.info("📤" + "="*50)

            order = self.session.place_order(**order_params)
            logger.info(f"📊 Ответ от биржи на ордер: {order}")

            order_id = order['result']['orderId']
            logger.info(f"✅ Ордер размещен: {order_id}")

            # Установка TP/SL
            if tp_percent > 0 or sl_percent > 0:
                tp_sl_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "takeProfit": str(tp_price),
                    "stopLoss": str(sl_price),
                    "positionIdx": position_index
                }
                
                logger.info("🛡️" + "="*50)
                logger.info("🛡️ ПАРАМЕТРЫ TP/SL:")
                logger.info(f"🛡️ Параметры TP/SL: {tp_sl_params}")
                logger.info("🛡️" + "="*50)

                try:
                    self.session.set_trading_stop(**tp_sl_params)
                    logger.info("✅ TP/SL установлены")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось установить TP/SL: {e}")

            final_result = {
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
            
            logger.info("✅" + "="*50)
            logger.info("✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
            logger.info(f"✅ {final_result}")
            logger.info("✅" + "="*50)

            return final_result

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
