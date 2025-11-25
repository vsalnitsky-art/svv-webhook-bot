"""
AI Analyst - Gemini Integration
Анализирует торговые сигналы через Google Gemini API
"""

import os
import logging

logger = logging.getLogger(__name__)

# Инициализация Gemini (ленивая загрузка)
_model = None
_api_configured = False

def _init_gemini():
    """Инициализация Gemini API"""
    global _model, _api_configured
    
    if _api_configured:
        return _model is not None
    
    try:
        import google.generativeai as genai
        
        # Получить API ключ из Environment Variable
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY не найден в Environment Variables")
            logger.info("Установите GEMINI_API_KEY в Render Dashboard → Environment")
            _api_configured = True
            return False
        
        # Настроить Gemini
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel('gemini-1.5-flash')
        _api_configured = True
        
        logger.info("✅ Gemini API успешно инициализирован")
        return True
        
    except ImportError:
        logger.error("❌ google-generativeai не установлен. Добавьте в requirements.txt")
        _api_configured = True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации Gemini: {e}")
        _api_configured = True
        return False


def analyze_signal(symbol, action, timeframe="15m"):
    """
    Анализ торгового сигнала через Gemini AI
    
    Args:
        symbol: Тикер монеты (например, BTCUSDT)
        action: Действие (Buy/Sell)
        timeframe: Таймфрейм (по умолчанию 15m)
    
    Returns:
        str: Анализ от AI или сообщение об ошибке
    """
    
    # Инициализация при первом вызове
    if not _init_gemini():
        return "⚠️ Gemini API не настроен. Установите GEMINI_API_KEY в Environment Variables."
    
    try:
        prompt = f"""
Ты профессиональный криптотрейдер с опытом 10+ лет.

Поступил торговый сигнал:
- Монета: {symbol}
- Действие: {action}
- Таймфрейм: {timeframe}

Дай краткий анализ (максимум 3-4 предложения):
1. Какой сейчас общий тренд на крипторынке?
2. Стоит ли доверять этому сигналу для {symbol}?
3. Укажи уровень риска: "РИСК НИЗКИЙ", "РИСК СРЕДНИЙ" или "РИСК ВЫСОКИЙ"
4. Одна конкретная рекомендация.

Будь конкретен и лаконичен. Не используй эмодзи.
"""
        
        response = _model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "⚠️ Gemini вернул пустой ответ"
            
    except Exception as e:
        logger.error(f"❌ Ошибка анализа Gemini: {e}")
        return f"Ошибка AI анализа: {str(e)[:100]}"


def get_market_sentiment():
    """
    Получить общее настроение рынка от Gemini
    
    Returns:
        str: Анализ настроения рынка
    """
    
    if not _init_gemini():
        return "Gemini API не настроен"
    
    try:
        prompt = """
Ты профессиональный криптоаналитик.

Дай краткую оценку текущего состояния криптовалютного рынка:
1. Общий тренд (бычий/медвежий/боковик)
2. Доминация Bitcoin и её влияние
3. Настроение рынка (страх/жадность)
4. Главные риски сейчас

Максимум 4-5 предложений. Будь конкретен.
"""
        
        response = _model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Не удалось получить анализ рынка"
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения sentiment: {e}")
        return f"Ошибка: {str(e)[:100]}"


# Тестовая функция
if __name__ == "__main__":
    # Тест AI модуля
    print("🧪 Тестирование AI Analyst...")
    
    result = analyze_signal("BTCUSDT", "Buy", "15m")
    print(f"\n📊 Результат анализа:\n{result}\n")
    
    sentiment = get_market_sentiment()
    print(f"\n📈 Настроение рынка:\n{sentiment}\n")