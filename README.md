# 🌙 Sleeper + Order Block Trading Bot

Автоматизований криптотрейдинг бот, який поєднує:
- **Sleeper Detection** - виявлення фаз накопичення
- **Order Block Analysis** - аналіз інституційних рівнів
- **Multi-timeframe Confirmation** - підтвердження на різних таймфреймах
- **Risk Management** - управління ризиками та позиціями

## 📋 Можливості

### Sleeper Detector v2.0
- 4H таймфрейм аналіз
- Multi-factor scoring:
  - Fuel Score (30%): funding rate + OI change
  - Volatility Score (25%): BB squeeze
  - Price Score (25%): range tightness
  - Liquidity Score (20%): volume profile
- HP система (0-10) для відстеження якості
- State machine: IDLE → WATCHING → BUILDING → READY → TRIGGERED

### Order Block Scanner
- Multi-timeframe: 15m → 5m → 1m
- Impulse detection (large body + high volume)
- Quality scoring 0-100
- MTF confirmation bonus
- Auto-expiry

### Trading
- Paper Trading з віртуальним балансом
- Live Trading через Bybit API
- Execution modes: Manual / Semi-Auto / Auto
- Position sizing за % ризику
- TP levels: 1R / 2R / 3R
- Trailing stop після 1.5R

## 🚀 Швидкий старт

### 1. Встановлення

```bash
# Клонувати репозиторій
git clone <repo-url>
cd sleeper_ob_bot

# Створити віртуальне середовище
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або
venv\Scripts\activate  # Windows

# Встановити залежності
pip install -r requirements.txt
```

### 2. Налаштування

Створити `.env` файл:

```env
# Bybit API (для live trading)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Telegram (опціонально)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Settings
PAPER_TRADING=true
EXECUTION_MODE=semi_auto
```

### 3. Запуск

```bash
# Повний режим (Web + Scheduler)
python main_bot.py

# Тільки Web UI
python main_bot.py --web-only

# Одноразовий скан
python main_bot.py --scan-only

# Ініціалізація БД
python main_bot.py --init-db
```

Відкрити в браузері: `http://localhost:5000`

## 📁 Структура проекту

```
sleeper_ob_bot/
├── config/
│   ├── bot_settings.py      # Enums, налаштування
│   └── bot_constants.py     # Пороги, константи
├── core/
│   ├── bybit_connector.py   # Bybit API client
│   ├── market_data.py       # Data fetcher + cache
│   └── tech_indicators.py   # RSI, ATR, BB (Wilder's)
├── detection/
│   ├── sleeper_scanner.py   # Sleeper Detector v2.0
│   ├── ob_scanner.py        # Order Block detector
│   └── signal_merger.py     # Signal integration
├── trading/
│   ├── risk_calculator.py   # Position sizing
│   ├── position_tracker.py  # P&L tracking
│   └── order_executor.py    # Paper/Live execution
├── storage/
│   ├── db_models.py         # SQLAlchemy models
│   └── db_operations.py     # CRUD operations
├── web/
│   └── flask_app.py         # Flask app + API
├── templates/               # HTML templates
├── static/                  # CSS + JS
├── alerts/
│   └── telegram_notifier.py # Telegram сповіщення
├── scheduler/
│   └── background_jobs.py   # Фонові задачі
├── main_bot.py              # Entry point
├── requirements.txt
├── render.yaml              # Render deployment
└── README.md
```

## 🖥️ Dashboard

| Сторінка | Опис |
|----------|------|
| Dashboard | Огляд, статистика, активні сигнали |
| Sleepers | Список Sleeper кандидатів |
| Order Blocks | Виявлені OB зони |
| Signals | Сигнали та підтвердження |
| Trades | Історія та відкриті позиції |
| Settings | Налаштування бота |

## ⚙️ API Endpoints

```
GET  /api/health              # Health check
GET  /api/stats               # Статистика
GET  /api/sleepers            # Список sleepers
GET  /api/orderblocks         # Order blocks
GET  /api/signals             # Сигнали
GET  /api/trades              # Трейди
GET  /api/positions           # Відкриті позиції
POST /api/scan/sleepers       # Запустити sleeper scan
POST /api/scan/orderblocks    # Запустити OB scan
POST /api/signal/confirm      # Підтвердити сигнал
POST /api/signal/reject       # Відхилити сигнал
POST /api/trade/close         # Закрити позицію
GET  /api/settings            # Отримати налаштування
POST /api/settings            # Оновити налаштування
```

## 📊 Risk Management

| Параметр | Значення |
|----------|----------|
| Risk per trade | 1-2% балансу |
| Max positions | 3 одночасно |
| Default leverage | 5x |
| TP1 | 1R (50% позиції) |
| TP2 | 2R (25% позиції) |
| TP3 | 3R (25% позиції) |
| Trailing start | 1.5R |
| Trailing offset | 0.5% |

## 🔔 Telegram сповіщення

Бот відправляє сповіщення про:
- Нові сигнали
- Відкриття/закриття позицій
- TP/SL hit
- Ready sleepers
- Якісні Order Blocks
- Системні події
- Денний звіт

## 🚀 Deployment на Render

1. Fork репозиторій на GitHub
2. Створити новий Web Service на Render
3. Підключити GitHub репозиторій
4. Додати Environment Variables
5. Deploy!

## ⚠️ Disclaimer

Цей бот призначений для освітніх цілей. Торгівля криптовалютами несе високий ризик. Використовуйте на власний ризик.

## 📝 License

MIT License
