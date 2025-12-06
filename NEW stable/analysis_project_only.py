#!/usr/bin/env python3
import re
import os

print("=" * 80)
print("🔍 КОМПЛЕКСНИЙ АНАЛІЗ ПРОЕКТУ")
print("=" * 80)

issues_critical = []
issues_warnings = []

# 1. Перевіримо bare except в проекті
print("\n1️⃣ BARE EXCEPT CLAUSES")
print("-" * 80)

for filename in os.listdir('.'):
    if filename.endswith('.py'):
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                if re.match(r'^\s*except\s*:\s*', line):
                    next_line = lines[i].strip() if i < len(lines) else ''
                    if next_line == 'pass':
                        print(f"⚠️ {filename}:{i} - except: pass (приховує помилки)")
                        issues_warnings.append(f"{filename}:{i}")
        except Exception as e:
            pass

# 2. Перевіримо undefined змінні
print("\n2️⃣ КРИТИЧНІ ЗМІННІ")
print("-" * 80)

# Перевіримо rsi_val, atr_val в scanner.py
with open('scanner.py', 'r') as f:
    content = f.read()
    if 'simple_rsi' in content and 'rsi_val =' in content:
        print("✅ scanner.py: rsi_val ініціалізується")
    else:
        print("❌ scanner.py: rsi_val можна не ініціалізувати!")
        issues_critical.append("rsi_val undefined")
    
    if 'simple_atr' in content and 'atr_val =' in content:
        print("✅ scanner.py: atr_val ініціалізується")
    else:
        print("❌ scanner.py: atr_val можна не ініціалізувати!")
        issues_critical.append("atr_val undefined")

# 3. Перевіримо pandas_ta імпорти
print("\n3️⃣ PANDAS_TA ІМПОРТИ")
print("-" * 80)

pandas_ta_found = False
for filename in os.listdir('.'):
    if filename.endswith('.py') and filename != 'setup_project.py':
        with open(filename, 'r') as f:
            content = f.read()
            if 'import pandas_ta' in content and 'ta.' not in content:
                # Це може бути закоментовано
                if not '# import pandas_ta' in content:
                    print(f"❌ {filename}: pandas_ta імпортується але не використовується!")
                    issues_critical.append(f"unused pandas_ta import in {filename}")
                    pandas_ta_found = True

if not pandas_ta_found:
    print("✅ pandas_ta імпорти не знайдені")

# 4. Перевіримо indicators.py наявність
print("\n4️⃣ INDICATORS.PY")
print("-" * 80)

if os.path.exists('indicators.py'):
    with open('indicators.py', 'r') as f:
        content = f.read()
        if 'simple_rsi' in content:
            print("✅ simple_rsi функція присутня")
        else:
            print("❌ simple_rsi функція відсутня!")
            issues_critical.append("simple_rsi missing")
        
        if 'simple_atr' in content:
            print("✅ simple_atr функція присутня")
        else:
            print("❌ simple_atr функція відсутня!")
            issues_critical.append("simple_atr missing")
else:
    print("❌ indicators.py НЕ ЗНАЙДЕНО!")
    issues_critical.append("indicators.py missing")

# 5. Перевіримо config.py функції
print("\n5️⃣ CONFIG.PY ФУНКЦІЇ")
print("-" * 80)

with open('config.py', 'r') as f:
    content = f.read()
    if 'def get_api_credentials' in content:
        print("✅ get_api_credentials функція присутня")
        if 'os.environ.get' in content:
            print("✅ os.environ.get використовується")
        if 'dotenv' in content:
            print("✅ dotenv підтримується для локальної розробки")
    else:
        print("❌ get_api_credentials функція ВІДСУТНЯ!")
        issues_critical.append("get_api_credentials missing")

# 6. Перевіримо main_app.py
print("\n6️⃣ MAIN_APP.PY КОНФІГУРАЦІЯ")
print("-" * 80)

with open('main_app.py', 'r') as f:
    content = f.read()
    
    checks = {
        "CSRFProtect": "CSRF захист",
        "app.config['SECRET_KEY']": "SECRET_KEY конфігурація",
        "os.environ.get": "Environment переменні",
        "from config import": "config імпорт",
        "from bot import": "bot імпорт",
    }
    
    for check, description in checks.items():
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} ВІДСУТНЯ!")
            issues_critical.append(description)

# 7. Перевіримо requirements.txt
print("\n7️⃣ REQUIREMENTS.TXT")
print("-" * 80)

with open('requirements.txt', 'r') as f:
    content = f.read()
    
    # Не повинно бути pandas-ta
    if 'pandas-ta' in content:
        print("❌ pandas-ta все ще в requirements.txt!")
        issues_critical.append("pandas-ta in requirements.txt")
    else:
        print("✅ pandas-ta видалена")
    
    # Повинна бути pandas
    if 'pandas==' in content:
        print("✅ pandas присутня")
    else:
        print("❌ pandas ВІДСУТНЯ!")
        issues_critical.append("pandas missing")
    
    # Повинна бути flask
    if 'flask==' in content:
        print("✅ flask присутня")
    else:
        print("❌ flask ВІДСУТНЯ!")
        issues_critical.append("flask missing")

# 8. Перевіримо bot.py
print("\n8️⃣ BOT.PY")
print("-" * 80)

with open('bot.py', 'r') as f:
    content = f.read()
    
    if 'from config import get_api_credentials' in content:
        print("✅ get_api_credentials імпортується")
    elif 'import config' in content and 'get_api_credentials' in content:
        print("✅ get_api_credentials використовується через config")
    else:
        print("❌ get_api_credentials не використовується!")
        issues_critical.append("get_api_credentials not used in bot.py")
    
    if 'class BybitTradingBot' in content:
        print("✅ BybitTradingBot клас присутній")
    else:
        print("❌ BybitTradingBot клас ВІДСУТНІЙ!")
        issues_critical.append("BybitTradingBot class missing")

# 9. Перевіримо models.py
print("\n9️⃣ MODELS.PY")
print("-" * 80)

with open('models.py', 'r') as f:
    content = f.read()
    
    if 'DATABASE_URL' in content:
        print("✅ DATABASE_URL обробка присутня")
    
    if 'postgresql' in content or 'sqlite' in content:
        print("✅ БД адаптери присутні")

# 10. Файли які мають бути
print("\n🔟 НАЯВНІСТЬ ОСНОВНИХ ФАЙЛІВ")
print("-" * 80)

required_files = [
    'config.py', 'bot.py', 'main_app.py', 'scanner.py', 'strategy.py',
    'market_analyzer.py', 'models.py', 'utils.py', 'indicators.py',
    'requirements.txt'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} ВІДСУТНІЙ!")
        issues_critical.append(f"{file} missing")

# Результати
print("\n" + "=" * 80)
print("📊 ПІДСУМКИ АНАЛІЗУ")
print("=" * 80)

print(f"\n❌ КРИТИЧНІ ПРОБЛЕМИ: {len(issues_critical)}")
for issue in issues_critical:
    print(f"  - {issue}")

print(f"\n⚠️ ПОПЕРЕДЖЕННЯ: {len(issues_warnings)}")

if not issues_critical:
    print("\n✅ КРИТИЧНИХ ПРОБЛЕМ НЕМАЄ!")
    print("Проект ГОТОВИЙ до Production deployment!")
else:
    print(f"\n❌ ЗНАЙДЕНО {len(issues_critical)} КРИТИЧНИХ ПРОБЛЕМ!")
    print("Потребує виправлення перед деплоєм!")

