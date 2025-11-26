# Руководство по установке 🚀

Подробные инструкции по установке и настройке SVV Webhook Trading Bot.

## Содержание

1. [Системные требования](#системные-требования)
2. [Локальная установка](#локальная-установка)
3. [Настройка API ключей](#настройка-api-ключей)
4. [Деплой на Render](#деплой-на-render)
5. [Первый запуск](#первый-запуск)
6. [Тестирование](#тестирование)
7. [Troubleshooting](#troubleshooting)

---

## Системные требования

### Минимальные требования
- Python 3.8 или выше
- pip (менеджер пакетов Python)
- 512 MB RAM
- 100 MB свободного места на диске

### Рекомендуемые требования
- Python 3.10+
- 1 GB RAM
- Стабильное интернет-соединение

### Поддерживаемые ОС
- ✅ Linux (Ubuntu 20.04+, Debian 10+)
- ✅ macOS (10.14+)
- ✅ Windows (10/11)

---

## Локальная установка

### Шаг 1: Клонирование репозитория

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/svv-webhook-bot.git

# Перейти в директорию
cd svv-webhook-bot
```

### Шаг 2: Создание виртуального окружения (рекомендуется)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Шаг 3: Установка зависимостей

```bash
pip install -r requirements.txt
```

**Список зависимостей:**
- Flask==3.0.0 - Веб-фреймворк
- pybit==5.6.2 - Bybit API клиент
- requests==2.31.0 - HTTP библиотека
- SQLAlchemy==2.0.23 - ORM для базы данных
- gunicorn==21.2.0 - WSGI сервер для production
- cryptography==41.0.7 - Шифрование API ключей
- google-generativeai - AI интеграция (опционально)

### Шаг 4: Проверка установки

```bash
python3 --version  # Должно быть 3.8+
pip list           # Проверить установленные пакеты
```

---

## Настройка API ключей

### Получение API ключей на Bybit

1. Зайдите на [Bybit](https://www.bybit.com)
2. Перейдите в **API Management**
3. Нажмите **Create New Key**
4. Настройте разрешения:
   - ✅ **Contract Trade** (Торговля)
   - ✅ **Position** (Позиции)
   - ✅ **Wallet** (Кошелек) - только для чтения
5. **Важно:** Сохраните API Key и Secret в безопасном месте!

### Вариант A: Незашифрованные ключи (для быстрого тестирования)

#### Linux/macOS:
```bash
export BYBIT_API_KEY="ваш_api_key"
export BYBIT_API_SECRET="ваш_api_secret"
```

#### Windows (PowerShell):
```powershell
$env:BYBIT_API_KEY="ваш_api_key"
$env:BYBIT_API_SECRET="ваш_api_secret"
```

#### Windows (CMD):
```cmd
set BYBIT_API_KEY=ваш_api_key
set BYBIT_API_SECRET=ваш_api_secret
```

### Вариант B: Зашифрованные ключи (рекомендуется для production)

#### Шаг 1: Генерация ключа шифрования

```bash
python3 -c "from config import generate_encryption_key; generate_encryption_key()"
```

**Вывод:**
```
============================================================
🔐 NEW ENCRYPTION KEY (Base64):
your_generated_encryption_key_here==
============================================================

Set this as ENCRYPTION_KEY in Render Environment Variables
```

**⚠️ ВАЖНО:** Сохраните этот ключ! Он понадобится для шифрования.

#### Шаг 2: Шифрование API ключей

```bash
python3 -c "from config import encrypt_credentials; encrypt_credentials('YOUR_API_KEY', 'YOUR_API_SECRET', 'YOUR_ENCRYPTION_KEY')"
```

**Вывод:**
```
============================================================
🔐 ENCRYPTED CREDENTIALS:
============================================================

BYBIT_API_KEY_ENCRYPTED=your_encrypted_api_key==

BYBIT_API_SECRET_ENCRYPTED=your_encrypted_api_secret==

============================================================
Copy these to Render Environment Variables
```

#### Шаг 3: Установка зашифрованных переменных

**Linux/macOS:**
```bash
export ENCRYPTION_KEY="your_encryption_key=="
export BYBIT_API_KEY_ENCRYPTED="your_encrypted_api_key=="
export BYBIT_API_SECRET_ENCRYPTED="your_encrypted_api_secret=="
```

**Windows (PowerShell):**
```powershell
$env:ENCRYPTION_KEY="your_encryption_key=="
$env:BYBIT_API_KEY_ENCRYPTED="your_encrypted_api_key=="
$env:BYBIT_API_SECRET_ENCRYPTED="your_encrypted_api_secret=="
```

### Вариант C: Файл .env (для локальной разработки)

1. Скопируйте пример:
```bash
cp .env.example .env
```

2. Отредактируйте `.env`:
```bash
nano .env  # или используйте любой текстовый редактор
```

3. Добавьте ваши ключи:
```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

4. **⚠️ ВАЖНО:** Никогда не коммитьте `.env` файл в Git!

---

## Деплой на Render

### Шаг 1: Подготовка репозитория

1. Создайте репозиторий на GitHub
2. Push ваш код:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Шаг 2: Создание Web Service на Render

1. Зайдите на [Render.com](https://render.com)
2. Нажмите **New +** → **Web Service**
3. Подключите ваш GitHub репозиторий
4. Настройки:
   - **Name:** `svv-webhook-bot` (или любое другое)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn main_app:app`
   - **Plan:** Free или платный

### Шаг 3: Настройка Environment Variables

В разделе **Environment** добавьте переменные:

**Вариант 1 (Незашифрованные):**
```
BYBIT_API_KEY = your_api_key
BYBIT_API_SECRET = your_api_secret
```

**Вариант 2 (Зашифрованные - Рекомендуется):**
```
ENCRYPTION_KEY = your_encryption_key
BYBIT_API_KEY_ENCRYPTED = your_encrypted_api_key
BYBIT_API_SECRET_ENCRYPTED = your_encrypted_api_secret
```

### Шаг 4: Деплой

1. Нажмите **Create Web Service**
2. Дождитесь завершения деплоя (2-5 минут)
3. Render автоматически назначит URL: `https://your-app.onrender.com`

### Шаг 5: Проверка

```bash
curl https://your-app.onrender.com/health
```

**Ожидаемый ответ:**
```json
{"status": "ok"}
```

---

## Первый запуск

### Локальный запуск

```bash
# Активировать виртуальное окружение (если используете)
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

# Запустить бота
python3 main_app.py
```

**Вывод:**
```
 * Serving Flask app 'main_app'
 * Debug mode: off
WARNING: This is a development server.
 * Running on http://0.0.0.0:10000
Press CTRL+C to quit
🚀 Position Monitor Started (Tracking active trades only)
```

### Проверка работы

1. Откройте браузер: `http://localhost:10000`
2. Вы будете перенаправлены на `/scanner`
3. Проверьте `/health` endpoint:
```bash
curl http://localhost:10000/health
```

---

## Тестирование

### Базовый тест подключения

```bash
# Использовать встроенный тестовый скрипт
python3 full_test.py
```

**Тесты проверяют:**
1. ✅ Здоровье сервера
2. ✅ Связь с Bybit API
3. ✅ Расчет торговой позиции

### Ручное тестирование webhook

```bash
curl -X POST http://localhost:10000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "Buy",
    "symbol": "BTCUSDT",
    "riskPercent": 1.0,
    "leverage": 5
  }'
```

**⚠️ ВАЖНО:** Это откроет реальную позицию! Используйте малые значения для тестирования.

### Тестирование на Testnet

Для использования Bybit Testnet измените в `bot.py`:

```python
# Было:
self.session = HTTP(testnet=False, api_key=k, api_secret=s)

# Стало:
self.session = HTTP(testnet=True, api_key=k, api_secret=s)
```

**Testnet API ключи:** Создайте отдельные ключи на [Bybit Testnet](https://testnet.bybit.com)

---

## Troubleshooting

### Проблема: "ModuleNotFoundError"

**Решение:**
```bash
# Убедитесь, что виртуальное окружение активировано
source venv/bin/activate

# Переустановите зависимости
pip install -r requirements.txt
```

### Проблема: "API credentials not found"

**Причина:** Переменные окружения не установлены

**Решение:**
```bash
# Проверьте переменные
echo $BYBIT_API_KEY  # Linux/macOS
echo %BYBIT_API_KEY%  # Windows CMD
$env:BYBIT_API_KEY   # Windows PowerShell

# Установите снова
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
```

### Проблема: "Port 10000 already in use"

**Решение:**
```bash
# Найти процесс
lsof -i :10000  # Linux/macOS
netstat -ano | findstr :10000  # Windows

# Убить процесс
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Или изменить порт в bot_config.py
PORT = 10001
```

### Проблема: Бот не открывает сделки

**Возможные причины:**
1. Недостаточно средств (минимум 5 USDT)
2. API ключи не имеют прав на торговлю
3. Неправильный формат symbol

**Решение:**
```bash
# Проверить баланс через Python
python3 -c "from bot import bot_instance; print(bot_instance.get_bal())"

# Проверить права API ключей на Bybit
# API Management → Edit → Убедитесь, что включено "Contract Trade"
```

### Проблема: "Decryption failed"

**Причина:** Неправильный ключ шифрования или зашифрованные ключи

**Решение:**
1. Проверьте ENCRYPTION_KEY
2. Перешифруйте ключи заново
3. Или используйте незашифрованные ключи для тестирования

### Проблема: Render деплой не работает

**Решение:**
1. Проверьте логи в Render Dashboard
2. Убедитесь, что `render.yaml` правильно настроен
3. Проверьте Environment Variables
4. Попробуйте Manual Deploy

---

## Следующие шаги

После успешной установки:

1. 📖 Прочитайте [API Documentation](API_DOCUMENTATION.md)
2. 🔧 Настройте параметры в `bot_config.py`
3. 📊 Изучите веб-интерфейсы `/scanner` и `/report`
4. 🔗 Настройте webhook в TradingView
5. 💡 Протестируйте на малых суммах

---

## Дополнительная помощь

- 📚 [README.md](README.md) - Общая документация
- 🔌 [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - API endpoints
- 📝 [CHANGELOG.md](CHANGELOG.md) - История изменений

**⚠️ Важно:** Всегда тестируйте бота на малых суммах перед использованием с крупными!

---

**🚀 Готово! Ваш бот установлен и готов к работе!**
