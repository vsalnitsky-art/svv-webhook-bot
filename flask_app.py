from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re

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

    def get_available_balance(self, currency="USDT"):
        """Получение доступного баланса в указанной валюте"""
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if balance_info.get('retCode') != 0:
                logger.error(f"❌ Ошибка API баланса: {balance_info.get('retMsg')}")
                return None
            
            result = balance_info.get('result', {})
            if not result or 'list' not in result or not result['list']:
                logger.error("❌ Пустой список в ответе баланса")
                return None
            
            account_list = result['list']
            
            # Ищем баланс в указанной валюте
            for account in account_list:
                # Проверяем монеты в аккаунте
                if 'coin' in account and account['coin']:
                    for coin in account['coin']:
                        if coin.get('coin') == currency:
                            coin_fields = [
                                'availableToWithdraw',
                                'walletBalance',
                                'equity',
                                'free'
                            ]
                            for field in coin_fields:
                                if field in coin:
                                    balance_str = coin[field]
                                    
                                    if balance_str and balance_str.strip() and balance_str not in ['', 'None', 'null']:
                                        try:
                                            clean_balance = str(balance_str).strip().replace(',', '').replace(' ', '')
                                            balance = float(clean_balance)
                                            if balance > 0:
                                                logger.info(f"💰 Найден {currency} баланс в поле {field}: ${balance}")
                                                return balance
                                        except (ValueError, TypeError) as e:
                                            logger.warning(f"⚠️ Ошибка конвертации {currency} {field}: {e}")
                                            continue
            
            logger.error(f"❌ Не удалось найти доступный баланс {currency}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return None

    def normalize_symbol(self, symbol):
        """Нормализация символа для Bybit API"""
        # Убираем .P и оставляем только базовый символ
        clean_symbol = symbol.replace('.P', '')
        logger.info(f"🔧 Нормализованный символ: {clean_symbol}")
        return clean_symbol

    def get_current_price(self, symbol):
        """Получение текущей цены"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            response = self.session.get_tickers(category="linear", symbol=normalized_symbol)
            
            if response.get('retCode') != 0:
                logger.error(f"❌ Ошибка API цены: {response.get('retMsg')}")
                return None
                
            result = response.get('result', {})
            tickers_list = result.get('list', [])
            
            if not tickers_list:
                logger.error("❌ Пустой список тикеров")
                return None
                
            ticker = tickers_list[0]
            last_price = ticker.get('lastPrice')
            
            if last_price:
                price = float(last_price)
                logger.info(f"💰 Текущая цена {normalized_symbol}: ${price}")
                return price
            
            logger.error("❌ Не удалось получить цену")
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения цены: {e}")
            return None

    def calculate_quantity(self, symbol, amount_usdt):
        """Правильный расчет количества на основе реальной цены"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            # Получаем текущую цену
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None, "Не удалось получить текущую цену"
            
            # Рассчитываем количество
            quantity = amount_usdt / current_price
            
            # Получаем информацию о символе для минимального количества и шага
            info = self.session.get_instruments_info(category="linear", symbol=normalized_symbol)
            if info and info.get('retCode') == 0:
                symbol_info = info['result']['list'][0]
                
                # Минимальное количество
                min_order_qty = float(symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 0))
                # Шаг количества
                qty_step = float(symbol_info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                # Минимальная сумма ордера
                min_order_value = float(symbol_info.get('lotSizeFilter', {}).get('minOrderAmt', 5))
                
                logger.info(f"📏 Параметры символа: minOrderQty={min_order_qty}, qtyStep={qty_step}, minOrderValue=${min_order_value}")
                
                # Проверяем минимальное количество
                if quantity < min_order_qty:
                    logger.warning(f"⚠️ Количество {quantity} меньше минимального {min_order_qty}")
                    return None, f"Слишком маленькая сумма для минимального объема. Минимум: ${min_order_value}"
                
                # Проверяем минимальную сумму
                order_value = quantity * current_price
                if order_value < min_order_value:
                    logger.warning(f"⚠️ Сумма ордера ${order_value} меньше минимальной ${min_order_value}")
                    return None, f"Сумма ордера меньше минимальной. Минимум: ${min_order_value}"
                
                # Округляем согласно шагу
                if qty_step > 0:
                    quantity = round(quantity // qty_step * qty_step, 8)
                
                logger.info(f"🔢 Количество от ${amount_usdt} по цене ${current_price}: {quantity}")
                return quantity, None
            else:
                # Fallback: простой расчет
                quantity = round(quantity, 6)
                logger.info(f"🔢 Количество от ${amount_usdt} (fallback): {quantity}")
                return quantity, None
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка расчета количества: {e}")
            return None, str(e)

    def place_order(self, data):
        try:
            # 📝 ЛОГИРОВАНИЕ ВХОДНЫХ ДАННЫХ
            logger.info("📥" + "="*50)
            logger.info("📥 ВХОДНЫЕ ДАННЫЕ ОТ TRADINGVIEW:")
            logger.info(f"📥 Тип данных: {type(data)}")
            logger.info(f"📥 Raw data: {data}")
            
            # Проверяем что данные не пустые
            if not data:
                return {"status": "error", "error": "Получены пустые данные"}
            
            # Принимаем параметры как есть из TradingView
            action = data.get('action')  # "Buy" или "Sell"
            symbol = data.get('symbol')  # Символ как есть (например "ADAUSDT.P")
            leverage = data.get('leverage', 25)  # Плечо (информационно)
            takeProfitPercent = data.get('takeProfitPercent', 0.5)  # TP в %
            stopLossPercent = data.get('stopLossPercent', 0.5)  # SL в %
            fixedAmount = data.get('fixedAmount', 6)  # Фиксированная сумма

            logger.info(f"📥 Action: {action}")
            logger.info(f"📥 Symbol: {symbol}")
            logger.info(f"📥 Leverage: {leverage}x (информационно)")
            logger.info(f"📥 TakeProfitPercent: {takeProfitPercent}%")
            logger.info(f"📥 StopLossPercent: {stopLossPercent}%")
            logger.info(f"📥 FixedAmount: ${fixedAmount}")
            logger.info("📥" + "="*50)

            # Валидация обязательных параметров
            if not action or not symbol:
                return {"status": "error", "error": "Отсутствуют обязательные параметры: action, symbol"}
            
            if action not in ['Buy', 'Sell']:
                return {"status": "error", "error": "Некорректное действие. Допустимо: Buy или Sell"}
            
            if fixedAmount <= 0:
                return {"status": "error", "error": "Фиксированная сумма должна быть больше 0"}

            # Используем USDT как валюту по умолчанию
            currency = "USDT"
            logger.info(f"💰 Валюта сделки: {currency}")

            # Получаем баланс в USDT
            real_balance = self.get_available_balance(currency)
            if real_balance is None:
                return {"status": "error", "error": f"Не удалось получить доступный баланс {currency}"}
            
            logger.info(f"💰 Реальный баланс {currency}: ${real_balance}")

            # Проверяем что фиксированная сумма не превышает баланс
            if fixedAmount > real_balance:
                return {"status": "error", "error": f"Недостаточно средств. Баланс: ${real_balance}, Запрошено: ${fixedAmount}"}

            # Нормализуем символ для Bybit API
            normalized_symbol = self.normalize_symbol(symbol)
            
            # Используем фиксированную сумму как есть
            position_amount = fixedAmount
            logger.info(f"💰 Фиксированная сумма: ${position_amount}")

            # ПРАВИЛЬНЫЙ расчет количества на основе реальной цены
            quantity, error = self.calculate_quantity(symbol, position_amount)
            if error:
                return {"status": "error", "error": error}

            # 📝 ЛОГИРОВАНИЕ РАСЧЕТОВ
            logger.info("🧮" + "="*50)
            logger.info("🧮 РАСЧЕТЫ ПОЗИЦИИ:")
            logger.info(f"🧮 Баланс {currency}: ${real_balance}")
            logger.info(f"🧮 Фиксированная сумма: ${position_amount}")
            logger.info(f"🧮 Плечо: {leverage}x (используется настройка биржи)")
            logger.info(f"🧮 Количество: {quantity}")
            logger.info("🧮" + "="*50)

            # Получаем текущую цену для TP/SL
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": f"Не удалось получить цену для {symbol}"}

            # Расчет TP/SL на основе реальной цены
            if action == "Buy":
                tp_price = round(current_price * (1 + takeProfitPercent / 100), 4) if takeProfitPercent > 0 else 0
                sl_price = round(current_price * (1 - stopLossPercent / 100), 4) if stopLossPercent > 0 else 0
                position_index = 0
            else:
                tp_price = round(current_price * (1 - takeProfitPercent / 100), 4) if takeProfitPercent > 0 else 0
                sl_price = round(current_price * (1 + stopLossPercent / 100), 4) if stopLossPercent > 0 else 0
                position_index = 1

            # 📝 ЛОГИРОВАНИЕ ТОРГОВЫХ ПАРАМЕТРОВ
            logger.info("🎯" + "="*50)
            logger.info("🎯 ТОРГОВЫЕ ПАРАМЕТРЫ:")
            logger.info(f"🎯 Действие: {action}")
            logger.info(f"🎯 Символ: {normalized_symbol}")
            logger.info(f"🎯 Плечо: {leverage}x (настройка биржи)")
            logger.info(f"🎯 Текущая цена: ${current_price}")
            logger.info(f"🎯 Сумма ордера: ${position_amount}")
            logger.info(f"🎯 Take Profit: ${tp_price} ({takeProfitPercent}%)" if tp_price > 0 else "🎯 Take Profit: не установлен")
            logger.info(f"🎯 Stop Loss: ${sl_price} ({stopLossPercent}%)" if sl_price > 0 else "🎯 Stop Loss: не установлен")
            logger.info(f"🎯 Количество: {quantity}")
            logger.info(f"🎯 Position Index: {position_index}")
            logger.info("🎯" + "="*50)

            # Размещение ордера
            order_params = {
                "category": "linear",
                "symbol": normalized_symbol,
                "side": action,
                "orderType": "Market",
                "qty": str(quantity),
                "timeInForce": "GTC",
            }
            
            logger.info("📤" + "="*50)
            logger.info("📤 ДАННЫЕ ДЛЯ ОТПРАВКИ НА БИРЖУ:")
            logger.info(f"📤 Параметры ордера: {order_params}")
            logger.info("📤" + "="*50)

            order = self.session.place_order(**order_params)
            logger.info(f"📊 Ответ от биржи на ордер: {order}")

            if order.get('retCode') != 0:
                error_msg = order.get('retMsg', 'Unknown error')
                logger.error(f"❌ Ошибка биржи: {error_msg}")
                return {"status": "error", "error": error_msg}

            order_id = order['result']['orderId']
            logger.info(f"✅ Ордер размещен: {order_id}")

            # Установка TP/SL если указаны проценты
            if takeProfitPercent > 0 or stopLossPercent > 0:
                tp_sl_params = {
                    "category": "linear",
                    "symbol": normalized_symbol,
                    "positionIdx": position_index
                }
                
                if takeProfitPercent > 0:
                    tp_sl_params["takeProfit"] = str(tp_price)
                if stopLossPercent > 0:
                    tp_sl_params["stopLoss"] = str(sl_price)
                
                logger.info("🛡️" + "="*50)
                logger.info("🛡️ ПАРАМЕТРЫ TP/SL:")
                logger.info(f"🛡️ Параметры TP/SL: {tp_sl_params}")
                logger.info("🛡️" + "="*50)

                try:
                    tp_sl_result = self.session.set_trading_stop(**tp_sl_params)
                    if tp_sl_result.get('retCode') == 0:
                        logger.info("✅ TP/SL установлены")
                    else:
                        logger.warning(f"⚠️ Не удалось установить TP/SL: {tp_sl_result.get('retMsg')}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось установить TP/SL: {e}")

            final_result = {
                "status": "success",
                "order_id": order_id,
                "symbol": normalized_symbol,
                "action": action,
                "quantity": float(quantity),
                "entry_price": current_price,
                "position_amount": position_amount,
                "currency": currency,
                "take_profit_price": tp_price,
                "stop_loss_price": sl_price,
                "take_profit_percent": takeProfitPercent,
                "stop_loss_percent": stopLossPercent,
                "leverage": leverage,
                "real_balance_used": real_balance,
                "note": "Плечо использует настройки биржи по умолчанию"
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

def parse_tradingview_data(raw_data):
    """Парсинг данных от TradingView"""
    try:
        logger.info(f"🔧 Парсинг сырых данных: {raw_data}")
        
        # Если данные уже словарь (приходят из request.get_json())
        if isinstance(raw_data, dict):
            logger.info("✅ Данные уже в формате словаря")
            return raw_data
        
        # Если это JSON строка
        if isinstance(raw_data, str):
            # Пробуем распарсить как чистый JSON
            try:
                data = json.loads(raw_data)
                logger.info("✅ Данные распарсены как JSON строка")
                return data
            except json.JSONDecodeError:
                pass
            
            # Если не получилось, ищем JSON-like структуру в тексте
            json_match = re.search(r'\{[^}]+\}', raw_data)
            if json_match:
                json_str = json_match.group()
                try:
                    data = json.loads(json_str)
                    logger.info("✅ JSON найден в тексте и распарсен")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Ошибка парсинга JSON из текста: {e}")
        
        logger.error("❌ Не удалось распарсить данные")
        return {}
        
    except Exception as e:
        logger.error(f"❌ Ошибка парсинга данных: {e}")
        return {}

@app.route('/')
def home():
    return "Trading Bot is Running! ✅ Use /webhook for TradingView"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Bot is running"})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Логируем все входящие данные для диагностики
        logger.info("🌐" + "="*50)
        logger.info("🌐 НОВЫЙ WEBHOOK ЗАПРОС")
        logger.info(f"🌐 Headers: {dict(request.headers)}")
        logger.info(f"🌐 Content-Type: {request.content_type}")
        logger.info(f"🌐 Method: {request.method}")
        
        # Получаем сырые данные
        raw_data = None
        if request.content_type and 'application/json' in request.content_type:
            raw_data = request.get_json()
            logger.info(f"🌐 Raw данные (JSON): {raw_data}")
        else:
            raw_data = request.get_data(as_text=True)
            logger.info(f"🌐 Raw данные (TEXT): {raw_data}")
        
        # Если raw_data None, пробуем получить данные из формы
        if raw_data is None:
            raw_data = request.form.to_dict()
            logger.info(f"🌐 Raw данные (FORM): {raw_data}")
        
        # Парсим данные от TradingView
        data = parse_tradingview_data(raw_data)
        logger.info(f"🌐 Данные после парсинга: {data}")
        
        if not data:
            logger.error("❌ Получены пустые данные после парсинга")
            return jsonify({"status": "error", "error": "No valid data received"}), 400
        
        # Обрабатываем ордер
        result = bot.place_order(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 Ошибка webhook: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
