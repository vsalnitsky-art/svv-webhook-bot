import google.generativeai as genai
# Импортируйте конфиг, если там лежит ключ, или вставьте напрямую ниже
from config import GEMINI_API_KEY 

# Настройка
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_signal(symbol, action, timeframe="15m"):
    """
    Функция спрашивает у Gemini мнение по тикеру
    """
    try:
        prompt = f"""
        Ты профессиональный крипто-трейдер.
        Поступил сигнал: {action} для монеты {symbol} (Таймфрейм: {timeframe}).
        
        Дай очень краткий анализ (максимум 3 предложения):
        1. Какой сейчас тренд у биткоина (общий фон)?
        2. Стоит ли доверять этому сигналу для {symbol}?
        3. Напиши "РИСК ВЫСОКИЙ" или "РИСК НИЗКИЙ".
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ошибка анализа ИИ: {e}"
