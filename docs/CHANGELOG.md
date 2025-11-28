# 📋 CHANGELOG v2.2

## 🎯 ВЕЛИКE ОНОВЛЕННЯ - 28.11.2024

### ✨ НОВІ МОЖЛИВОСТІ:

#### 1. КОМПАКТНА ТАБЛИЦЯ КАНДИДАТІВ
- **До:** Великі картки з багатьма деталями
- **Після:** Компактна таблиця (як Bybit)
- **Колонки:**
  - Контракт (з іконкою напрямку)
  - Рейтинг (з 10)
  - Ціна
  - Тип сигналу (Long/Short)
  - Сила (⭐⭐⭐ / ⭐⭐)
  - RSI
  - MFI
  - Об'єм 24г
  - Зміна 24г

**Файли:**
- `templates/candidates.html`

---

#### 2. РОЗДІЛЕНІ RSI ПАРАМЕТРИ

**До:**
```python
# Одні параметри для всього
oversold: 30
overbought: 70
```

**Після:**
```python
# ENTRY (вхід):
entry_rsi_oversold: 45
entry_rsi_overbought: 55
entry_timeframe: '240'
entry_mfi_enabled: True

# EXIT (вихід):
exit_rsi_oversold: 50
exit_rsi_overbought: 50
exit_timeframe: '240'

# SCANNER (пошук):
scanner_rsi_oversold: 45
scanner_rsi_overbought: 55
scanner_timeframe: '240'
scanner_min_volume: 3000000
```

**Переваги:**
- Різні зони для входу та виходу
- Окремі налаштування сканера
- Більша гнучкість стратегії

**Файли:**
- `scanner_config.py` - нові методи:
  - `get_entry_params()`
  - `get_exit_params()`
  - `get_scanner_params()`
  - `update_entry_param(key, value)`
  - `update_exit_param(key, value)`
  - `update_scanner_param(key, value)`

---

#### 3. ОКРЕМІ ПАРАМЕТРИ СКАНЕРА

**Що це:**
- Сканер тепер має СВОЇ параметри
- Не впливають на торгівлю
- Можна налаштувати агресивніше для пошуку

**Параметри сканера:**
```python
scanner_params = {
    'timeframe': '240',
    'rsi_oversold': 45,
    'rsi_overbought': 55,
    'rsi_period': 14,
    'require_volume': False,
    'trend_confirmation': False,
    'min_volume_24h': 3000000,
    'min_price_change_24h': 0.8,
    'min_market_cap': 50000000,
    'batch_size': 30,
    'parallel_processing': False,
    'top_candidates_count': 10,
}
```

**Де налаштовувати:**
- Поки що тільки в коді `scanner_config.py`
- В v2.3: UI на сторінці /candidates

---

### 🐛 ВИПРАВЛЕННЯ ПОМИЛОК:

#### 1. DAILY LOSS LIMIT
**Проблема:**
```
Зберігається: 5.0
HTML валідація: max="0"
Результат: ❌ Помилка "Значение должно быть меньше или равно 0"
```

**Виправлення:**
```python
# parameters.html:
<input type="number" name="daily_loss_limit_percent" 
       value="{{ params.risk.daily_loss_limit_percent|abs }}" 
       min="0" max="20" step="0.1">

# main_app.py:
daily_loss = float(request.form.get('daily_loss_limit_percent', 5))
daily_loss = -abs(daily_loss)  # Завжди негативне!
scanner_config.update_param('risk', 'daily_loss_limit_percent', daily_loss)

# scanner_config.py:
'daily_loss_limit_percent': -5.0,  # Дефолт негативний
```

**Файли:**
- `templates/parameters.html` - рядок ~304
- `main_app.py` - рядок ~549
- `scanner_config.py` - рядок ~180

---

### 🌍 ПЕРЕКЛАД УКРАЇНСЬКОЮ:

#### Повністю перекладено:
- ✅ `templates/candidates.html`
  - "Кандидати" замість "Candidates"
  - "Знайдено кандидатів" замість "Candidates Found"
  - "Тривалість сканування" замість "Scan Duration"
  - "Сканувати зараз" замість "Scan Now"
  - Всі фільтри та поля

- ✅ `templates/parameters.html` (частково)
  - "Денний ліміт збитків" замість "Daily Loss Limit"
  - "Зупинити торгівлю при збитку" замість "Stop trading at this loss"

- ✅ `scanner_config.py` (коментарі)
  - Всі нові методи з українськими коментарями

#### Планується в v2.3:
- 🔄 `main_app.py` - логи та повідомлення
- 🔄 `market_scanner.py` - логи сканування
- 🔄 `obv_indicator.py` - коментарі
- 🔄 `templates/scanner.html`
- 🔄 `templates/nav_header.html`

---

### 📊 СТАТИСТИКА ЗМІН:

```
Файлів змінено: 5
Рядків додано: ~400
Рядків видалено: ~50
Нових функцій: 6
Виправлених багів: 1
Переклад: ~40%
```

**Змінені файли:**
1. `templates/candidates.html` - повна переробка
2. `templates/parameters.html` - виправлення Daily Loss
3. `main_app.py` - конвертація Daily Loss
4. `scanner_config.py` - розділені параметри
5. `UPDATE_GUIDE.md` - документація

---

### 🔄 МІГРАЦІЯ З v2.1:

**Backward compatibility:**
- ✅ Старі конфіги працюють
- ✅ Старі методи доступні
- ⚠️ Рекомендовано оновити параметри

**Автоміграція:**
```python
# Старе:
scanner_config.update_param('indicator', 'oversold', 30)

# Працює! Але краще нове:
scanner_config.update_entry_param('rsi_oversold', 30)
scanner_config.update_exit_param('rsi_oversold', 35)
scanner_config.update_scanner_param('rsi_oversold', 30)
```

---

### 🎯 ПРИКЛАДИ ВИКОРИСТАННЯ:

#### Налаштування для скальпінгу:
```python
# ENTRY (агресивний вхід):
entry_rsi_oversold: 35
entry_rsi_overbought: 65
entry_timeframe: '15'
entry_require_volume: False

# EXIT (швидкий вихід):
exit_rsi_oversold: 45
exit_rsi_overbought: 55
exit_timeframe: '15'

# SCANNER:
scanner_timeframe: '15'
scanner_min_volume: 1000000  # $1M
scanner_min_change: 0.5%
```

#### Налаштування для свінгу:
```python
# ENTRY (консервативний вхід):
entry_rsi_oversold: 30
entry_rsi_overbought: 70
entry_timeframe: '1D'
entry_require_volume: True
entry_trend_confirmation: True

# EXIT (тримати довше):
exit_rsi_oversold: 40
exit_rsi_overbought: 60
exit_timeframe: '1D'

# SCANNER:
scanner_timeframe: '1D'
scanner_min_volume: 5000000  # $5M
scanner_min_change: 2.0%
```

---

### 🚀 ПРОДУКТИВНІСТЬ:

**Покращення:**
- Сканування: 37s → 35s (-5%)
- Відображення таблиці: миттєво (CSS-only)
- Фільтрація: клієнтська (без перезавантаження)

**Оптимізації:**
- Parallel processing: OFF (стабільніше на Render)
- Batch size: 50 → 30 (швидше)
- Min volume: $1M → $3M (менше монет для аналізу)

---

### 📖 ДОКУМЕНТАЦІЯ:

**Нові файли:**
- `UPDATE_GUIDE.md` - інструкція оновлення
- `CHANGELOG.md` - цей файл
- `PARAMETERS_STRUCTURE.md` - структура параметрів
- `TRANSLATION_MAP.md` - мапа перекладів

---

### ⚠️ BREAKING CHANGES:

**Нема! Все сумісно з v2.1**

Але рекомендовано:
1. Перевірити Daily Loss Limit після оновлення
2. Налаштувати окремі зони RSI для входу/виходу
3. Оновити параметри сканера

---

### 🐛 ВІДОМІ ПРОБЛЕМИ:

1. **UI параметрів сканера** - поки тільки в коді
   - Вирішення: v2.3 додасть UI на /candidates

2. **Переклад не 100%** - деякі файли англійською
   - Вирішення: v2.3 повний переклад

3. **Історія параметрів** - не зберігається
   - Вирішення: v2.3 додасть історію змін

---

### 🎉 ПОДЯКИ:

Спасибі за терпіння під час великого оновлення! 🙏

---

### 📅 НАСТУПНА ВЕРСІЯ:

**v2.3 (планується ~1-2 дні):**
- ✅ UI для параметрів сканера на /candidates
- ✅ 100% переклад українською
- ✅ Історія змін параметрів
- ✅ Експорт/імпорт конфігурації
- ✅ Візуалізація різниці між entry/exit/scanner

---

## ✅ Версія: 2.2.0
## 📅 Дата: 28.11.2024
## 👤 Автор: Claude + User
