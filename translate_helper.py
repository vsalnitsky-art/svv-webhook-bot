#!/usr/bin/env python3
"""
Масовий переклад коментарів та рядків українською
"""

TRANSLATIONS = {
    # Загальні терміни
    "Trading Style": "Стиль торгівлі",
    "Aggressiveness": "Агресивність",
    "Automation Mode": "Режим автоматизації",
    "Indicator Parameters": "Параметри індикатора",
    "Risk Management": "Управління ризиками",
    "Scanner": "Сканер",
    "Timeframe": "Таймфрейм",
    
    # Стилі торгівлі
    "Scalping": "Скальпінг",
    "Day Trading": "Денна торгівля",
    "Swing Trading": "Свінг-торгівля",
    
    # Параметри
    "Default": "За замовчуванням",
    "Enable": "Увімкнути",
    "Disable": "Вимкнути",
    "Enabled": "Увімкнено",
    "Disabled": "Вимкнено",
    
    # RSI/MFI
    "Oversold": "Перепроданість",
    "Overbought": "Перекупленість",
    "RSI Period": "Період RSI",
    "MFI Period": "Період MFI",
    
    # Ризики
    "Max Positions": "Макс позицій",
    "Position Size": "Розмір позиції",
    "Daily Loss Limit": "Денний ліміт збитків",
    "Leverage": "Кредитне плече",
    
    # Дії
    "Save": "Зберегти",
    "Reset": "Скинути",
    "Apply": "Застосувати",
    "Cancel": "Скасувати",
    "Scan Now": "Сканувати зараз",
    
    # Статуси
    "Active": "Активний",
    "Inactive": "Неактивний",
    "Running": "Виконується",
    "Stopped": "Зупинено",
    "Success": "Успішно",
    "Failed": "Помилка",
    "Error": "Помилка",
    
    # Повідомлення
    "Settings saved successfully": "Налаштування успішно збережено",
    "Parameters updated": "Параметри оновлено",
    "Scan completed": "Сканування завершено",
    "No candidates found": "Кандидатів не знайдено",
}

def translate_text(text):
    """Перекласти текст"""
    for en, uk in TRANSLATIONS.items():
        text = text.replace(en, uk)
    return text

if __name__ == '__main__':
    print("Словник перекладів підготовлено")
    print(f"Всього термінів: {len(TRANSLATIONS)}")
