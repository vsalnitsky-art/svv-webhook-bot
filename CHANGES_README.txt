Змінені файли (гілка claude/analyze-project-...)
================================================
Архів містить ПОТОЧНІ версії всіх файлів, що відрізняються від базової
точки гілки. Структура папок збережена — копіюйте поверх проєкту.

НОВА ФІЧА — Fuel Auto-Filter:
  detection/fuel_filter.py      — новий daemon (паливо→таймер→угода)
  web/flask_app.py              — init + /api/fuel-filter/* ендпоінти
  templates/smart_money.html    — панель ❤️ Fuel Auto-Filter + ❤️ у watchlist

ВИДАЛЕНО (Trend-Exhaustion 4H/1H + Trade Quality Gate):
  detection/auto_gate.py        — прибрано Telegram trend-alerts
  detection/smc_scanner.py      — прибрано get_bias()
  detection/trade_manager.py    — прибрано QG-оцінку + blocked trades
  storage/db_models.py          — прибрано модель BlockedTrade
  templates/smart_money.html    — прибрано панелі Виснаження тренду,
                                  Quality Gate, Blocked Trades, колонку QG
  web/flask_app.py              — прибрано reversal-розрахунок,
                                  /api/blocked-trades/*
  ВИДАЛЕНІ ФАЙЛИ (їх немає в архіві — видаліть у себе вручну):
    detection/reversal_pressure.py
    detection/quality_gate_v2.py

ІНШІ ЗМІНИ (з попередніх комітів гілки):
  .gitignore, templates/base.html, templates/dashboard.html,
  templates/database_admin.html, test_toggle_strategy.py

Примітка: storage/db_operations.py НЕ в архіві — його зміни (методи
blocked-trades) були додані й знову видалені, тож відносно бази він
не змінився.
