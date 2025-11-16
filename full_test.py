import requests
import json

# Ваш URL на Render
BASE_URL = "https://svv-webhook-bot.onrender.com"

def run_tests():
    print("🧪 ЗАПУСК ТЕСТОВ ПОДКЛЮЧЕНИЯ")
    print("=" * 50)
    
    # Тест 1: Health check
    print("1. Проверка здоровья сервера...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("   ✅ Сервер работает")
        else:
            print(f"   ❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Не удалось подключиться: {e}")
        return
    
    # Тест 2: Проверка связи с Bybit
    print("2. Проверка связи с Bybit...")
    try:
        response = requests.get(f"{BASE_URL}/test-connection")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Связь с Bybit установлена")
            print(f"   💰 Цена BTC: {data.get('btc_price', 'N/A')}")
        else:
            print(f"   ❌ Ошибка связи с Bybit: {response.text}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 3: Тест расчета позиции
    print("3. Тест расчета торговой позиции...")
    try:
        test_data = {
            "action": "BUY",
            "symbol": "BTCUSDT",
            "leverage": 5,
            "riskPercent": 1,
            "accountBalance": 1000,
            "takeProfitPercent": 3,
            "stopLossPercent": 1.5
        }
        
        response = requests.post(f"{BASE_URL}/test-trading", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Расчет успешен")
            print(f"   📊 Размер позиции: ${data['position_size_usdt']}")
            print(f"   📈 TP: ${data['take_profit_price']}")
            print(f"   📉 SL: ${data['stop_loss_price']}")
        else:
            print(f"   ❌ Ошибка расчета: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("=" * 50)
    print("🎯 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    run_tests()
