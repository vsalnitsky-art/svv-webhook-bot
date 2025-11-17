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

    def calculate_position_size_usdt(self, balance, risk_percent, leverage):
        """Расчет размера позиции ТОЛЬКО в USDT"""
        position_size_usdt = balance * (risk_percent / 100) * leverage
        return round(position_size_usdt, 2)

    def get_available_balance(self):
        """Получение доступного баланса в USDT"""
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
            
            # Ищем USDT баланс
            for account in account_list:
                # Пробуем разные поля с балансом
                balance_fields = [
                    'totalAvailableBalance',
                    'totalWalletBalance', 
                    'totalEquity',
                    'totalMarginBalance'
                ]
                
                for field in balance_fields:
                    if field in account:
                        balance_str = account[field]
                        
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
        """Получение текущей цены"""
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
            
            if response.get('retCode') != 0:
                logger.error(f"❌ Ошибка API: {response.get('retMsg')}")
                return None
                
            result = response.get('result', {})
            tickers_list = result.get('list', [])
            
            if not tickers_list:
                logger.error("❌ Пустой список тикеров")
                return None
                
            ticker = tickers_list[0]
            
            price_fields = ['lastPrice', 'markPrice', 'indexPrice', 'prevPrice24h']
            
            for field in price_fields:
                price_str = ticker.get(field)
                if price_str:
                    try:
                        clean_price = str(price_str).strip().replace(',', '').replace(' ', '')
                        if clean_price and clean_price not in ['', 'None', 'null', '0']:
                            price = float(clean_price)
                            if price > 0:
                                logger.info(f"💰 Цена {symbol}: ${price}")
                                return price
                    except (ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Ошибка конвертации {field}: {e}")
                        continue
            
            logger.error("❌ Ни одно поле цены не содержит валидных данных")
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
            logger.info(f"✅ Плечо установлено: {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Не удалось установить плечо: {e}")
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

    def calculate_quantity_from_usdt(self, symbol, usdt_amount, price):
        """Расчет количества от суммы в USDT"""
        try:
            # Получаем информацию о символе для минимального количества и шага
            info = self.session.get_instruments_info(category="linear", symbol=symbol)
            if info and info.get('retCode') == 0:
                symbol_info = info['result']['list'][0]
                min_order_qty = float(symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 0))
                qty_step = float(symbol_info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                
                logger.info(f"📏 Параметры символа: minOrderQty={min_order_qty}, qtyStep={qty_step}")
                
                # Рассчитываем количество от USDT
                quantity = usdt_amount / price
                
                # Проверяем минимальное количество
                if quantity < min_order_qty:
                    logger.warning(f"⚠️ Количество {quantity} меньше минимального {min_order_qty}")
                    # Увеличиваем сумму чтобы достичь минимума
                    min_usdt = min_order_qty * price
                    if min_usdt > usdt_amount:
                        logger.info(f"💰 Увеличиваем сумму до минимума: ${min_usdt}")
                        usdt_amount = min_usdt
                        quantity = min_order_qty
                
                # Округляем согласно шагу
                if qty_step > 0:
                    quantity = round(quantity // qty_step * qty_step, 8)
                
                logger.info(f"🔢 Количество от ${usdt_amount}: {quantity}")
                return quantity, usdt_amount
                
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить параметры символа: {e}")
        
        # Fallback: простой расчет
        quantity = usdt_amount / price
        quantity = round(quantity, 6)  # Округляем до 6 знаков
        
        logger.info(f"🔢 Количество от ${usdt_amount} (fallback): {quantity}")
        return quantity, usdt_amount

    def place_order(self, data):
        try:
            # 📝 ЛОГИРОВАНИЕ ВХОДНЫХ ДАННЫХ
            logger.info("📥" + "="*50)
            logger.info("📥 ВХОДНЫЕ ДАННЫЕ ОТ TRADINGVIEW:")
            logger.info(f"📥 Raw data: {data}")
            
            action = data.get('action', 'Buy')
            raw_symbol = data.get('symbol', 'BTCUSDT')
            leverage = min(data.get('leverage', 5), 25)
            risk_percent = min(data.get('riskPercent', 1), 10)
            tp_percent = data.get('takeProfitPercent', 3)
            sl_percent = data.get('stopLossPercent', 1.5)
            fixed_amount = data.get('fixedAmount', 0)  # Фиксированная сумма в USDT

            logger.info(f"📥 Action: {action}")
            logger.info(f"📥 Symbol: {raw_symbol}")
            logger.info(f"📥 Leverage: {leverage}")
            logger.info(f"📥 RiskPercent: {risk_percent}")
            logger.info(f"📥 TakeProfitPercent: {tp_percent}")
            logger.info(f"📥 StopLossPercent: {sl_percent}")
            logger.info(f"📥 FixedAmount: {fixed_amount}")
            logger.info("📥" + "="*50)

            # Нормализация символа
            symbol = self.normalize_symbol(raw_symbol)

            # Получаем баланс
            real_balance = self.get_available_balance()
            if real_balance is None:
                return {"status": "error", "error": "Не удалось получить доступный баланс"}
            
            logger.info(f"💰 Реальный баланс: ${real_balance}")

            # Получение цены
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"status": "error", "error": f"Не удалось получить цену для {symbol}"}

            # РАСЧЕТ СУММЫ В USDT
            if fixed_amount and fixed_amount > 0:
                # Используем фиксированную сумму в USDT
                position_size_usdt = min(fixed_amount, real_balance)
                logger.info(f"💰 Фиксированная сумма: ${fixed_amount}")
                logger.info(f"💰 Используется: ${position_size_usdt}")
            else:
                # Расчет от баланса и риска в USDT
                position_size_usdt = self.calculate_position_size_usdt(
                    real_balance, risk_percent, leverage
                )
                logger.info(f"💰 Рассчитанная сумма: ${position_size_usdt}")

            # Рассчитываем количество от суммы в USDT
            quantity, final_usdt_amount = self.calculate_quantity_from_usdt(
                symbol, position_size_usdt, current_price
            )

            # Проверка минимального объема
            if quantity <= 0:
                return {"status": "error", "error": f"Слишком маленький объем: {quantity}"}

            # 📝 ЛОГИРОВАНИЕ РАСЧЕТОВ
            logger.info("🧮" + "="*50)
            logger.info("🧮 РАСЧЕТЫ ПОЗИЦИИ В USDT:")
            logger.info(f"🧮 Баланс: ${real_balance}")
            if fixed_amount:
                logger.info(f"🧮 Фиксированная сумма: ${fixed_amount}")
            else:
                logger.info(f"🧮 Риск: {risk_percent}%")
                logger.info(f"🧮 Плечо: {leverage}x")
            logger.info(f"🧮 Цена: ${current_price}")
            logger.info(f"🧮 Сумма позиции (USDT): ${final_usdt_amount}")
            logger.info(f"🧮 Количество (монет): {quantity}")
            logger.info("🧮" + "="*50)

            # Установка плеча (игнорируем ошибки)
            self.set_leverage(symbol, leverage)

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
            logger.info(f"🎯 Сумма ордера: ${final_usdt_amount}")
            logger.info(f"🎯 Take Profit: ${tp_price} ({tp_percent}%)")
            logger.info(f"🎯 Stop Loss: ${sl_price} ({sl_percent}%)")
            logger.info(f"🎯 Количество: {quantity}")
            logger.info(f"🎯 Position Index: {position_index}")
            logger.info("🎯" + "="*50)

            # Размещение ордера
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": action,
                "orderType": "Market",
                "qty": str(quantity),
                "timeInForce": "GTC",
            }
            
            logger.info("📤" + "="*50)
            logger.info("📤 ДАННЫЕ ДЛЯ ОТПРАВКИ НА БИРЖИ:")
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

            # Установка TP/SL
            if tp_percent > 0 or sl_percent > 0:
                tp_sl_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "takeProfit": str(tp_price),
                    "stopLoss": str(sl_price),
                    "positionIdx": position_index
                }
                
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
                "quantity": float(quantity),
                "entry_price": current_price,
                "position_size_usdt": final_usdt_amount,
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

def parse_tradingview_data(raw_data):
    """Парсинг данных от TradingView (plain text -> JSON)"""
    try:
        logger.info(f"🔧 Парсинг сырых данных: {raw_data}")
        
        # Если это JSON строка
        if isinstance(raw_data, str) and raw_data.strip().startswith('{'):
            try:
                # Пробуем распарсить как JSON
                data = json.loads(raw_data)
                logger.info("✅ Данные распарсены как JSON")
                return data
            except json.JSONDecodeError:
                # Если не JSON, ищем JSON-like структуру в тексте
                pass
        
        # Если это plain text, ищем JSON-like структуру
        if isinstance(raw_data, str):
            # Ищем что-то похожее на JSON в тексте
            json_match = re.search(r'\{[^}]+\}', raw_data)
            if json_match:
                json_str = json_match.group()
                try:
                    data = json.loads(json_str)
                    logger.info("✅ JSON найден в тексте и распарсен")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Ошибка парсинга JSON из текста: {e}")
        
        # Если ничего не нашли, создаем базовую структуру из текста
        if isinstance(raw_data, str):
            data = {}
            if "BUY" in raw_data.upper() or "LONG" in raw_data.upper():
                data['action'] = 'Buy'
            elif "SELL" in raw_data.upper() or "SHORT" in raw_data.upper():
                data['action'] = 'Sell'
            
            # Пробуем найти символ в тексте
            symbol_match = re.search(r'([A-Z0-9]+\.?[A-Z0-9]*)', raw_data)
            if symbol_match:
                data['symbol'] = symbol_match.group(1)
            
            logger.info(f"🔄 Данные извлечены из текста: {data}")
            return data
        
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
        if request.content_type and 'application/json' in request.content_type:
            raw_data = request.get_json()
            logger.info(f"🌐 Raw данные (JSON): {raw_data}")
        else:
            raw_data = request.get_data(as_text=True)
            logger.info(f"🌐 Raw данные (TEXT): {raw_data}")
        
        # Парсим данные от TradingView
        data = parse_tradingview_data(raw_data)
        logger.info(f"🌐 Данные после парсинга: {data}")
        
        if not data:
            return jsonify({"status": "error", "error": "No valid data received"}), 400
        
        # Обрабатываем ордер
        result = bot.place_order(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 Ошибка webhook: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
