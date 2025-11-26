# Руководство по безопасности 🔐

Важная информация о безопасном использовании SVV Webhook Trading Bot.

## 🚨 Критически важно

### Никогда не делайте это:
- ❌ НЕ коммитьте API ключи в Git
- ❌ НЕ публикуйте API ключи в публичных репозиториях
- ❌ НЕ передавайте API ключи по незащищенным каналам
- ❌ НЕ используйте одни и те же ключи для разных ботов
- ❌ НЕ давайте API ключам больше прав, чем необходимо
- ❌ НЕ используйте Master API ключи для ботов

---

## 🔑 Управление API ключами

### Создание безопасных API ключей на Bybit

1. **Минимальные права:**
   - ✅ Contract Trade (только для торговли)
   - ✅ Position (управление позициями)
   - ✅ Wallet (только READ для проверки баланса)
   - ❌ Withdrawal (НЕ включайте!)

2. **IP Whitelist:**
   - Добавьте только IP адреса вашего сервера
   - Для Render: узнайте выходной IP
   - Для TradingView: `52.89.214.238`, `34.212.75.30`, `54.218.53.128`, `52.32.178.7`

3. **Срок действия:**
   - Используйте временные ключи если возможно
   - Меняйте ключи каждые 3-6 месяцев

### Хранение ключей

#### ✅ Правильно:
```bash
# Переменные окружения
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."

# Или зашифрованные
export ENCRYPTION_KEY="..."
export BYBIT_API_KEY_ENCRYPTED="..."
export BYBIT_API_SECRET_ENCRYPTED="..."
```

#### ❌ Неправильно:
```python
# Хардкод в коде
API_KEY = "abc123..."  # НИКОГДА ТАК НЕ ДЕЛАЙТЕ!

# Текстовый файл в репозитории
with open('keys.txt') as f:  # ОПАСНО!
    api_key = f.read()
```

---

## 🔒 Шифрование

### Использование Fernet для защиты ключей

```bash
# 1. Сгенерировать ключ шифрования (один раз)
python3 -c "from config import generate_encryption_key; generate_encryption_key()"

# 2. Зашифровать ваши API ключи
python3 -c "from config import encrypt_credentials; encrypt_credentials('YOUR_API_KEY', 'YOUR_API_SECRET', 'YOUR_ENCRYPTION_KEY')"

# 3. Использовать зашифрованные ключи в production
```

### Безопасное хранение ключа шифрования

- Храните ENCRYPTION_KEY отдельно от зашифрованных ключей
- Используйте менеджеры секретов (AWS Secrets Manager, HashiCorp Vault)
- Для Render: используйте Environment Variables (они не видны в логах)

---

## 🌐 Безопасность веб-приложения

### HTTPS

**Production:**
- ✅ Всегда используйте HTTPS
- ✅ Render автоматически предоставляет SSL сертификат
- ❌ Никогда не используйте HTTP для production webhook

**Development:**
- HTTP допустим только для локального тестирования
- Используйте ngrok для тестирования с TradingView

### Защита endpoints

```python
# Пример добавления простой авторизации
from flask import request, abort

WEBHOOK_TOKEN = os.environ.get('WEBHOOK_TOKEN')

@app.route('/webhook', methods=['POST'])
def webhook():
    # Проверка токена
    token = request.headers.get('Authorization')
    if token != f'Bearer {WEBHOOK_TOKEN}':
        abort(401)
    
    # Ваша логика...
```

### Rate Limiting

Рекомендуется добавить ограничение частоты запросов:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)

@app.route('/webhook', methods=['POST'])
@limiter.limit("10 per minute")
def webhook():
    # Ваша логика...
```

---

## 💾 Безопасность базы данных

### SQLite

```python
# ✅ Правильно: относительный путь, не в публичной директории
db_path = 'trading_bot_final.db'

# ❌ Неправильно: абсолютный путь или публичная директория
db_path = '/var/www/html/database.db'  # ОПАСНО!
```

### Резервное копирование

```bash
# Регулярное резервное копирование
cp trading_bot_final.db backups/db_$(date +%Y%m%d).db

# Автоматическое резервное копирование (cron)
0 0 * * * cp /path/to/trading_bot_final.db /path/to/backups/db_$(date +\%Y\%m\%d).db
```

### Шифрование базы данных (опционально)

Для дополнительной безопасности можно использовать SQLCipher:

```bash
pip install sqlcipher3
```

```python
from sqlalchemy import create_engine

db_password = os.environ.get('DB_PASSWORD')
engine = create_engine(
    f'sqlite+pysqlcipher://:{db_password}@/trading_bot.db?cipher=aes-256-cfb'
)
```

---

## 🚫 Защита от атак

### Валидация входных данных

```python
def validate_webhook_data(data):
    """Проверка данных от webhook"""
    
    # Проверка обязательных полей
    if 'action' not in data or 'symbol' not in data:
        raise ValueError("Missing required fields")
    
    # Проверка action
    if data['action'] not in ['Buy', 'Sell']:
        raise ValueError("Invalid action")
    
    # Проверка symbol
    if not re.match(r'^[A-Z]{3,10}USDT$', data['symbol']):
        raise ValueError("Invalid symbol format")
    
    # Проверка числовых значений
    if 'riskPercent' in data:
        risk = float(data['riskPercent'])
        if risk < 0.1 or risk > 100:
            raise ValueError("Risk percent out of range")
    
    return True
```

### Защита от SQL injection

SQLAlchemy ORM автоматически защищает от SQL injection, но:

```python
# ✅ Правильно: использование ORM
session.query(Trade).filter_by(symbol=symbol).all()

# ❌ Неправильно: прямой SQL
session.execute(f"SELECT * FROM trades WHERE symbol = '{symbol}'")  # ОПАСНО!
```

### Защита от XSS

```python
from flask import escape

# ✅ Правильно: экранирование пользовательского ввода
safe_text = escape(user_input)

# Jinja2 автоматически экранирует переменные:
{{ variable }}  # Безопасно
{{ variable|safe }}  # Использовать ТОЛЬКО для доверенного контента
```

---

## 📊 Мониторинг безопасности

### Логирование подозрительной активности

```python
import logging

# Логировать неудачные попытки
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        validate_webhook_data(data)
    except Exception as e:
        logger.warning(f"Suspicious webhook: {request.remote_addr} - {e}")
        return jsonify({"error": "invalid"}), 400
```

### Алерты

Настройте уведомления для:
- Неудачные попытки аутентификации
- Необычно большие сделки
- Быстрое уменьшение баланса
- Ошибки API

---

## 🔍 Аудит безопасности

### Регулярные проверки

**Еженедельно:**
- ✅ Проверьте активные API ключи на Bybit
- ✅ Проверьте логи на подозрительную активность
- ✅ Проверьте баланс и открытые позиции

**Ежемесячно:**
- ✅ Обновите зависимости: `pip install --upgrade -r requirements.txt`
- ✅ Проверьте CVE для используемых библиотек
- ✅ Проверьте права доступа к файлам
- ✅ Сделайте резервную копию базы данных

**Ежеквартально:**
- ✅ Смените API ключи
- ✅ Обновите ENCRYPTION_KEY (если используется)
- ✅ Проведите тест восстановления из резервной копии

### Проверка зависимостей

```bash
# Проверить уязвимости в зависимостях
pip install safety
safety check

# Или использовать
pip-audit
```

---

## 🚨 Реагирование на инциденты

### Если API ключи скомпрометированы:

1. **Немедленно:**
   - Удалите скомпрометированные ключи в Bybit
   - Создайте новые ключи
   - Обновите переменные окружения

2. **Проверьте:**
   - История сделок на необычную активность
   - Изменения баланса
   - Логи доступа

3. **Уведомите:**
   - Bybit поддержку (если есть подозрительные транзакции)
   - Членов команды

### Если обнаружена уязвимость:

1. Отключите бота
2. Обновите код/зависимости
3. Проверьте базу данных на целостность
4. Перезапустите с новыми ключами

---

## 📋 Чеклист безопасности

Перед запуском в production:

- [ ] API ключи зашифрованы или в переменных окружения
- [ ] IP Whitelist настроен на Bybit
- [ ] HTTPS включен (Render делает это автоматически)
- [ ] Логирование настроено
- [ ] .gitignore настроен (файл .env не в репозитории)
- [ ] Базовая валидация входных данных реализована
- [ ] Минимальные права для API ключей
- [ ] Резервное копирование БД настроено
- [ ] Мониторинг настроен
- [ ] План реагирования на инциденты готов

---

## 📚 Дополнительные ресурсы

### Документация
- [Bybit API Security](https://bybit-exchange.github.io/docs/v5/guide#authentication)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Flask Security](https://flask.palletsprojects.com/en/2.3.x/security/)

### Инструменты
- [Safety](https://pypi.org/project/safety/) - проверка уязвимостей Python пакетов
- [Bandit](https://github.com/PyCQA/bandit) - статический анализ безопасности Python
- [pip-audit](https://pypi.org/project/pip-audit/) - аудит зависимостей

---

## 🤝 Сообщить об уязвимости

Если вы обнаружили уязвимость:
1. **НЕ** публикуйте детали публично
2. Свяжитесь с владельцем репозитория напрямую
3. Предоставьте детальное описание

---

**⚠️ Помните: Безопасность - это не одноразовая настройка, а постоянный процесс!**

**🔐 Безопасной торговли!**
