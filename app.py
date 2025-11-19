from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
from config import get_api_credentials

# Налаштування
PORT = 10000  # Порт винесено в змінну для keep_alive та app.run

def keep_alive():
    """Функція для підтримки сервера активним (Dynamic Localhost)"""
    time.sleep(5) # Чекаємо трохи, поки сервер запуститься
    while True:
        try:
            # Використовуємо локальний інтерфейс, щоб не залежати від домену
            requests.get(f'http://127.0.0.1:{PORT}/health', timeout=5)
            print("🔄 Keep-alive ping sent (localhost)")
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600) # Пінг кожні 10 хвилин

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Запуск keep-alive в окремому потоці
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
            logger.info("✅ Бот ініціалізований із зашифрованими ключами")
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації бота: {e}")
            raise

    def get_available_balance(self, currency="USDT"):
        """Отримання доступного балансу"""
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if balance_info.get('retCode') != 0:
                logger.error(f"❌ Помилка API балансу: {balance_info.get('retMsg')}")
                return None
            
            result = balance_info.get('result', {})
            if not result or 'list' not in result or not result['list']:
                logger.error("❌ Порожній список у відповіді балансу")
                return None
            
            account_list = result['list']
            
            for account in account_list:
                if 'coin' in account and account['coin']:
                    for coin in account['coin']:
                        if coin.get('coin') == currency:
                            # Логіка пошуку доступного балансу по пріоритету полів
                            coin_fields = ['availableToWithdraw', 'walletBalance', 'equity', 'free']
                            for field in coin_fields:
                                if field in coin:
                                    balance_str = coin[field]
                                    if balance_str and balance_str.strip() and balance_str not in ['', 'None', 'null']:
                                        try:
                                            clean_balance = str(balance_str).strip().replace(',', '').replace(' ', '')
                                            balance = float(clean_balance)
                                            if balance > 0:
                                                logger.info(f"💰 Знайдено {currency} баланс ({field}): ${balance}")
                                                return balance
                                        except (ValueError, TypeError):
                                            continue
            
            logger.error(f"❌ Не вдалося знайти доступний баланс {currency}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Помилка отримання балансу: {e}")
            return None

    def normalize_symbol(self, symbol):
        """Нормалізація символу (прибираємо .P)"""
        clean_symbol = symbol.replace('.P', '')
        # logger.info(f"🔧 Нормалізований символ: {clean_symbol}") # Можна розкоментувати для дебагу
        return clean_symbol

    def get_current_price(self, symbol):
        """Отримання поточної ціни"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            response = self.session.get_tickers(category="linear", symbol=normalized_symbol)
            
            if response.get('retCode') != 0:
                return None
                
            result = response.get('result', {})
            tickers_list = result.get('list', [])
            
            if tickers_list:
                ticker = tickers_list[0]
                last_price = ticker.get('lastPrice')
                if last_price:
                    return float(last_price)
            return None
            
        except Exception as e:
            logger.error(f"❌ Помилка отримання ціни: {e}")
            return None

    def set_leverage(self, symbol, leverage):
        """Встановлення кредитного плеча"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            leverage_str = str(leverage)
            
            logger.info(f"⚙️ Спроба встановити плече {leverage_str}x для {normalized_symbol}...")
            
            response = self.session.set_leverage(
                category="linear",
                symbol=normalized_symbol,
                buyLeverage=leverage_str,
                sellLeverage=leverage_str
            )
            
            if response.get('retCode') == 0:
                logger.info(f"✅ Плече {leverage_str}x успішно встановлено")
                return True
            else:
                # Якщо плече вже встановлено таким же (код помилки 110043), це не помилка
                if response.get('retCode') == 110043:
                    logger.info(f"ℹ️ Плече вже дорівнює {leverage_str}x (пропускаємо)")
                    return True
                
                logger.warning(f"⚠️ Не вдалося встановити плече: {response.get('retMsg')}")
                return False
                
        except Exception as e:
            # Обробка виключення, якщо бібліотека кидає помилку при retCode != 0
            if "110043" in str(e):
                logger.info(f"ℹ️ Плече вже встановлено (API Exception caught)")
                return True
            logger.error(f"❌ Помилка при зміні плеча: {e}")
            return False

    def calculate_quantity(self, symbol, amount_usdt):
        """Розрахунок кількості монет"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                return None, "Не вдалося отримати поточну ціну"
            
            quantity = amount_usdt / current_price
            
            # Отримання інформації про інструмент для округлення
            info = self.session.get_instruments_info(category="linear", symbol=normalized_symbol)
            if info and info.get('retCode') == 0:
                symbol_info = info['result']['list'][0]
                
                min_order_qty = float(symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 0))
                qty_step = float(symbol_info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                min_order_value = float(symbol_info.get('lotSizeFilter', {}).get('minOrderAmt', 5))
                
                if quantity < min_order_qty:
                    return None, f"Занадто мала сума. Мінімум: {min_order_qty} монет"
                
                if (quantity * current_price) < min_order_value:
                    return None, f"Сума ордера менша за мінімальну ${min_order_value}"
                
                # Округлення
                if qty_step > 0:
                    # Використовуємо decimal або строкове форматування для точності, тут спрощено
                    import decimal
                    step_decimals = abs(decimal.Decimal(str(qty_step)).as_tuple().exponent)
                    quantity = round(quantity // qty_step * qty_step, step_decimals)
                
                return quantity, None
            else:
                return round(quantity, 6), None
                
        except Exception as e:
            return None, str(e)

    def set_tp_sl(self, symbol, action, current_price, takeProfitPercent, stopLossPercent):
        """Встановлення TP/SL"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            if action == "Buy":
                tp_price = round(current_price * (1 + takeProfitPercent / 100), 4) if takeProfitPercent > 0 else 0
                sl_price = round(current_price * (1 - stopLossPercent / 100), 4) if stopLossPercent > 0 else 0
                position_index = 0 # 0 для One-Way Mode
            else:
                tp_price = round(current_price * (1 - takeProfitPercent / 100), 4) if takeProfitPercent > 0 else 0
                sl_price = round(current_price * (1 + stopLossPercent / 100), 4) if stopLossPercent > 0 else 0
                position_index = 0

            tp_sl_params = {
                "category": "linear",
                "symbol": normalized_symbol,
                "positionIdx": position_index
            }
            
            if takeProfitPercent > 0: tp_sl_params["takeProfit"] = str(tp_price)
            if stopLossPercent > 0: tp_sl_params["stopLoss"] = str(sl_price)
            
            logger.info(f"🛡️ Встановлення TP: {tp_price} | SL: {sl_price}")

            tp_sl_result = self.session.set_trading_stop(**tp_sl_params)
            
            if tp_sl_result.get('retCode') == 0:
                logger.info("✅ TP/SL успішно встановлені")
                return True
            else:
                logger.warning(f"⚠️ Помилка TP/SL (Main): {tp_sl_result.get('retMsg')}")
                # Альтернативний метод без positionIdx
                try:
                    alt_params = {"category": "linear", "symbol": normalized_symbol}
                    if takeProfitPercent > 0: alt_params["takeProfit"] = str(tp_price)
                    if stopLossPercent > 0: alt_params["stopLoss"] = str(sl_price)
                    
                    alt_result = self.session.set_trading_stop(**alt_params)
                    if alt_result.get('retCode') == 0:
                        logger.info("✅ TP/SL встановлені (Alt method)")
                        return True
                except:
                    pass
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Виключення при TP/SL: {e}")
            return False

    def place_order(self, data):
        try:
            # 1. Валідація
            if not data: return {"status": "error", "error": "Empty data"}
            
            action = data.get('action')
            symbol = data.get('symbol')
            leverage = data.get('leverage', 20) # Дефолтне плече
            takeProfitPercent = data.get('takeProfitPercent', 0.0)
            stopLossPercent = data.get('stopLossPercent', 0.0)
            
            # Отримуємо відсоток ризику, за замовчуванням 5%
            riskPercent = float(data.get('riskPercent', 5.0))

            if not action or not symbol:
                return {"status": "error", "error": "Missing action or symbol"}
            
            # 2. Баланс
            currency = "USDT"
            real_balance = self.get_available_balance(currency)
            
            if real_balance is None:
                return {"status": "error", "error": f"Could not get balance for {currency}"}

            # 3. ВСТАНОВЛЕННЯ ПЛЕЧА
            normalized_symbol = self.normalize_symbol(symbol)
            self.set_leverage(normalized_symbol, leverage)

            # ==============================================================================
            # 4. ДИНАМІЧНИЙ РОЗРАХУНОК РОЗМІРУ ПОЗИЦІЇ
            # Алгоритм: Balance * (Risk% / 100) * Leverage
            # ==============================================================================
            
            # а) Вираховуємо маржу (скільки грошей беремо з балансу)
            margin_amount = real_balance * (riskPercent / 100.0)
            
            # б) Вираховуємо повний розмір позиції з урахуванням плеча
            total_position_value_usdt = margin_amount * leverage
            
            logger.info("🧮" + "="*50)
            logger.info("🧮 РОЗРАХУНОК РОЗМІРУ ОРДЕРА (DYNAMIC):")
            logger.info(f"🧮 Доступний баланс: ${real_balance}")
            logger.info(f"🧮 Відсоток ризику: {riskPercent}%")
            logger.info(f"🧮 Використана маржа: ${margin_amount:.2f}")
            logger.info(f"🧮 Плече: {leverage}x")
            logger.info(f"🧮 ЗАГАЛЬНА СУМА ОРДЕРА: ${total_position_value_usdt:.2f}")
            logger.info("🧮" + "="*50)

            if margin_amount > real_balance:
                 return {"status": "error", "error": f"Margin ${margin_amount} > Balance ${real_balance}"}

            # 5. Розрахунок кількості монет
            # Передаємо повну суму позиції в calculate_quantity
            quantity, error = self.calculate_quantity(symbol, total_position_value_usdt)
            if error: return {"status": "error", "error": error}

            current_price = self.get_current_price(symbol)
            
            # 6. Розміщення ордера
            logger.info(f"🚀 Відкриття {action} {normalized_symbol} x{leverage} | Qty: {quantity}")
            
            order_params = {
                "category": "linear",
                "symbol": normalized_symbol,
                "side": action,
                "orderType": "Market",
                "qty": str(quantity),
                "timeInForce": "GTC",
            }
            
            order = self.session.place_order(**order_params)
            
            if order.get('retCode') != 0:
                return {"status": "error", "error": order.get('retMsg')}

            order_id = order['result']['orderId']
            logger.info(f"✅ Ордер успішний: {order_id}")

            # 7. TP/SL
            tp_sl_success = False
            if takeProfitPercent > 0 or stopLossPercent > 0:
                tp_sl_success = self.set_tp_sl(symbol, action, current_price, takeProfitPercent, stopLossPercent)

            return {
                "status": "success",
                "order_id": order_id,
                "symbol": normalized_symbol,
                "leverage": leverage,
                "used_margin": margin_amount,
                "total_value": total_position_value_usdt,
                "tp_sl_set": tp_sl_success
            }

        except Exception as e:
            logger.error(f"❌ Критична помилка ордера: {e}")
            return {"status": "error", "error": str(e)}

# Ініціалізація бота
bot = BybitTradingBot()

def parse_tradingview_data(raw_data):
    """Парсинг даних (JSON/String)"""
    try:
        if isinstance(raw_data, dict): return raw_data
        if isinstance(raw_data, str):
            import json
            import re
            try:
                return json.loads(raw_data)
            except:
                json_match = re.search(r'\{[^}]+\}', raw_data)
                if json_match:
                    return json.loads(json_match.group())
        return {}
    except:
        return {}

@app.route('/')
def home():
    return "🤖 Trading Bot Active"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw_data = None
        if request.content_type and 'application/json' in request.content_type:
            raw_data = request.get_json()
        else:
            raw_data = request.get_data(as_text=True) or request.form.to_dict()
        
        data = parse_tradingview_data(raw_data)
        
        if not data:
            return jsonify({"status": "error", "error": "No data"}), 400
            
        result = bot.place_order(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 Webhook error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    # Цей блок виконується тільки при локальному запуску через python app.py
    # Gunicorn ігнорує цей блок і імпортує app напряму
    logger.info(f"🚀 Запуск Development сервера на порту {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
