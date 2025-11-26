# API Documentation 📡

Полная документация API endpoints для SVV Webhook Trading Bot.

## Base URL

```
Production: https://your-app.onrender.com
Local: http://localhost:10000
```

## Endpoints

### 1. Health Check

Проверка работоспособности сервера.

**Endpoint:** `GET /health`

**Ответ:**
```json
{
  "status": "ok"
}
```

**Пример:**
```bash
curl https://your-app.onrender.com/health
```

---

### 2. Webhook (Торговые сигналы)

Основной endpoint для получения торговых сигналов от TradingView или других источников.

**Endpoint:** `POST /webhook`

**Content-Type:** `application/json`

#### Параметры запроса

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `action` | string | ✅ | Направление сделки: "Buy" или "Sell" |
| `symbol` | string | ✅ | Торговая пара (например, "BTCUSDT", "ETHUSDT") |
| `riskPercent` | float | ❌ | Процент риска от баланса (по умолчанию: 5.0) |
| `leverage` | integer | ❌ | Кредитное плечо (по умолчанию: 20) |
| `takeProfit` | float | ❌ | Цена тейк-профита (абсолютное значение) |
| `takeProfitPercent` | float | ❌ | Тейк-профит в процентах (например, 3.0 = +3%) |
| `stopLoss` | float | ❌ | Цена стоп-лосса (абсолютное значение) |
| `stopLossPercent` | float | ❌ | Стоп-лосс в процентах (например, 1.5 = -1.5%) |

#### Примеры запросов

**Минимальный запрос:**
```json
{
  "action": "Buy",
  "symbol": "BTCUSDT"
}
```

**Полный запрос:**
```json
{
  "action": "Buy",
  "symbol": "BTCUSDT",
  "riskPercent": 5.0,
  "leverage": 20,
  "takeProfitPercent": 3.0,
  "stopLossPercent": 1.5
}
```

**С абсолютными ценами:**
```json
{
  "action": "Sell",
  "symbol": "ETHUSDT",
  "riskPercent": 3.0,
  "leverage": 15,
  "takeProfit": 2800.00,
  "stopLoss": 3100.00
}
```

#### Ответы

**Успех:**
```json
{
  "status": "ok"
}
```

**Ошибка:**
```json
{
  "error": "error"
}
```
HTTP Status: 400

**Недостаточно средств:**
```json
{
  "status": "no_balance"
}
```

#### cURL примеры

```bash
# Базовый запрос
curl -X POST https://your-app.onrender.com/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "Buy",
    "symbol": "BTCUSDT"
  }'

# Полный запрос
curl -X POST https://your-app.onrender.com/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "Buy",
    "symbol": "BTCUSDT",
    "riskPercent": 5.0,
    "leverage": 20,
    "takeProfitPercent": 3.0
  }'
```

---

### 3. Scanner (Мониторинг позиций)

Веб-интерфейс для мониторинга активных позиций в реальном времени.

**Endpoint:** `GET /scanner`

**Отображает:**
- Активные позиции с P&L
- RSI (Индекс относительной силы)
- Рыночное давление (Volume Pressure)
- Последние логи мониторинга
- Историю закрытых сделок

**Автообновление:** Страница обновляется каждые 5 секунд

**Доступ через браузер:**
```
https://your-app.onrender.com/scanner
```

---

### 4. Report (Аналитика P&L)

Детальная аналитика торговой производительности.

**Endpoint:** `GET /report`

#### Query параметры

| Параметр | Тип | Описание | Пример |
|----------|-----|----------|--------|
| `days` | integer | Период в днях | `?days=7` |
| `start` | date | Дата начала (YYYY-MM-DD) | `?start=2024-01-01` |
| `end` | date | Дата окончания (YYYY-MM-DD) | `?end=2024-01-31` |

#### Примеры

```bash
# За последние 7 дней
https://your-app.onrender.com/report?days=7

# За последние 30 дней
https://your-app.onrender.com/report?days=30

# За конкретный период
https://your-app.onrender.com/report?start=2024-01-01&end=2024-01-31
```

**Отображаемые данные:**
- Общий P&L за период
- Win rate (процент успешных сделок)
- График изменения баланса
- Топ-5 монет по прибыльности
- P&L по Long и Short позициям
- Детальная таблица всех сделок

---

## Интеграция с TradingView

### Настройка Alert в TradingView

1. Откройте график в TradingView
2. Создайте новый Alert (иконка будильника)
3. В настройках Alert:
   - **Условие:** Выберите ваше условие (например, пересечение MA)
   - **Webhook URL:** `https://your-app.onrender.com/webhook`
   - **Message:** Используйте JSON формат

### Пример Alert Message для TradingView

**Long сигнал:**
```json
{
  "action": "Buy",
  "symbol": "{{ticker}}",
  "riskPercent": 5.0,
  "leverage": 20,
  "takeProfitPercent": 3.0
}
```

**Short сигнал:**
```json
{
  "action": "Sell",
  "symbol": "{{ticker}}",
  "riskPercent": 5.0,
  "leverage": 20,
  "takeProfitPercent": 3.0
}
```

**С динамическими переменными TradingView:**
```json
{
  "action": "Buy",
  "symbol": "{{ticker}}",
  "riskPercent": 5.0,
  "leverage": 20,
  "takeProfit": {{close}},
  "note": "Signal at {{time}} on {{interval}}"
}
```

---

## Логика торговли

### Расчет размера позиции

```
Размер позиции = (Баланс × RiskPercent / 100 × Leverage × 0.98) / Текущая_цена
```

Где:
- `Баланс` - Доступный USDT баланс
- `RiskPercent` - Процент риска (по умолчанию 5%)
- `Leverage` - Кредитное плечо (по умолчанию 20)
- `0.98` - Коэффициент для учета комиссий
- `Текущая_цена` - Актуальная рыночная цена

### Трейлинг стоп-лосс

Автоматически устанавливается в зависимости от актива:

| Актив | Трейлинг % |
|-------|-----------|
| BTC, ETH, BNB | 0.5% |
| SOL, XRP, ADA | 1.5% |
| Остальные | 3.0% |

### Split Take Profit

Позиция автоматически разделяется на два тейк-профита:

1. **50% позиции** закрывается на 40% пути к целевой цене
2. **50% позиции** закрывается на целевой цене

Пример для BUY на $50,000 с TP $51,500:
- TP1: $50,600 (40% от $1,500 = $600)
- TP2: $51,500 (целевая цена)

---

## Коды ответов

| Код | Статус | Описание |
|-----|--------|----------|
| 200 | OK | Запрос успешно обработан |
| 400 | Bad Request | Некорректный формат запроса |
| 500 | Internal Server Error | Внутренняя ошибка сервера |

---

## Ограничения

### Rate Limits
- Webhook: Без жестких ограничений, но рекомендуется не более 1 запроса в секунду
- Scanner/Report: Нет ограничений

### Минимальные требования
- Минимальный баланс для торговли: **5 USDT**
- Минимальный размер ордера: Зависит от актива (определяется биржей)

---

## Безопасность

### Headers
Все запросы должны содержать:
```
Content-Type: application/json
```

### HTTPS
В продакшене рекомендуется использовать **только HTTPS**.

### IP Whitelist
Рекомендуется настроить IP whitelist в Bybit API настройках:
- TradingView IPs: `52.89.214.238`, `34.212.75.30`, `54.218.53.128`, `52.32.178.7`

---

## Мониторинг

### Health Check
Используйте `/health` endpoint для мониторинга:

```bash
# Проверка каждые 5 минут
*/5 * * * * curl -f https://your-app.onrender.com/health || exit 1
```

### Логи
Все важные события логируются:
- Открытие позиций: `✅ OPEN: Buy BTCUSDT`
- Закрытие позиций: Записывается в базу данных
- Ошибки: `Order Error: [описание]`

---

## Примеры интеграций

### Python
```python
import requests
import json

def send_signal(action, symbol, risk=5.0):
    url = "https://your-app.onrender.com/webhook"
    data = {
        "action": action,
        "symbol": symbol,
        "riskPercent": risk,
        "leverage": 20,
        "takeProfitPercent": 3.0
    }
    
    response = requests.post(url, json=data)
    return response.json()

# Использование
result = send_signal("Buy", "BTCUSDT", 5.0)
print(result)
```

### JavaScript (Node.js)
```javascript
const axios = require('axios');

async function sendSignal(action, symbol, risk = 5.0) {
  const url = 'https://your-app.onrender.com/webhook';
  const data = {
    action: action,
    symbol: symbol,
    riskPercent: risk,
    leverage: 20,
    takeProfitPercent: 3.0
  };
  
  try {
    const response = await axios.post(url, data);
    return response.data;
  } catch (error) {
    console.error('Error:', error.message);
    return null;
  }
}

// Использование
sendSignal('Buy', 'BTCUSDT', 5.0)
  .then(result => console.log(result));
```

### PHP
```php
<?php
function sendSignal($action, $symbol, $risk = 5.0) {
    $url = 'https://your-app.onrender.com/webhook';
    $data = array(
        'action' => $action,
        'symbol' => $symbol,
        'riskPercent' => $risk,
        'leverage' => 20,
        'takeProfitPercent' => 3.0
    );
    
    $options = array(
        'http' => array(
            'header'  => "Content-type: application/json\r\n",
            'method'  => 'POST',
            'content' => json_encode($data)
        )
    );
    
    $context  = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    return json_decode($result, true);
}

// Использование
$result = sendSignal('Buy', 'BTCUSDT', 5.0);
print_r($result);
?>
```

---

## Troubleshooting

### Ошибка "error" в ответе
**Причины:**
- Неверный формат JSON
- Некорректные параметры
- Ошибка подключения к Bybit

**Решение:**
- Проверьте формат запроса
- Убедитесь, что symbol существует на Bybit
- Проверьте логи сервера

### Ошибка "no_balance"
**Причина:** Недостаточно средств на балансе (< 5 USDT)

**Решение:**
- Пополните баланс
- Уменьшите leverage или riskPercent

### Сделка не открывается
**Возможные причины:**
- API ключи не имеют прав на торговлю
- Позиция по этому активу уже открыта
- Неправильный symbol (проверьте формат: "BTCUSDT")

**Решение:**
- Проверьте права API ключей в Bybit
- Закройте существующую позицию
- Используйте корректный формат символа

---

## Обновления API

**Версия:** 1.0.0

**Последнее обновление:** Ноябрь 2024

**Изменения:**
- Добавлена поддержка split take profit
- Улучшена логика трейлинг стоп-лосса
- Добавлена расширенная аналитика

---

**⚡ Для дополнительной информации см. README.md**
