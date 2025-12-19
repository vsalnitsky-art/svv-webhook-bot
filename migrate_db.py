#!/usr/bin/env python3
"""
Міграція бази даних - додавання нових колонок до SmartMoneyExecutionLog та DetectedOrderBlock
"""
import sqlite3
import sys

def migrate(db_path='trading_bot_final.db'):
    print(f"🔄 Міграція БД: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Нові колонки для smart_money_execution_log
    new_columns_exec = [
        ("ob_type", "VARCHAR(10)"),
        ("ob_start_time", "DATETIME"),
        ("ob_midline", "FLOAT"),
        ("ob_size_percent", "FLOAT"),
    ]
    
    # Нові колонки для detected_order_blocks
    new_columns_ob = [
        ("ob_start_time", "DATETIME"),
        ("ob_midline", "FLOAT"),
        ("ob_size_percent", "FLOAT"),
    ]
    
    # Додаємо колонки до smart_money_execution_log
    print("\n📋 smart_money_execution_log:")
    for col_name, col_type in new_columns_exec:
        try:
            cursor.execute(f"ALTER TABLE smart_money_execution_log ADD COLUMN {col_name} {col_type}")
            print(f"  ✅ Додано: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print(f"  ⏭️  Вже існує: {col_name}")
            else:
                print(f"  ❌ Помилка: {col_name} - {e}")
    
    # Додаємо колонки до detected_order_blocks
    print("\n📋 detected_order_blocks:")
    for col_name, col_type in new_columns_ob:
        try:
            cursor.execute(f"ALTER TABLE detected_order_blocks ADD COLUMN {col_name} {col_type}")
            print(f"  ✅ Додано: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print(f"  ⏭️  Вже існує: {col_name}")
            else:
                print(f"  ❌ Помилка: {col_name} - {e}")
    
    conn.commit()
    conn.close()
    
    print("\n✅ Міграція завершена!")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'trading_bot_final.db'
    migrate(db_path)
