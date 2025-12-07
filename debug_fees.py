#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug скрипт для перевірки структури даних від Bybit API
Показує які поля приходять для закритих угод
"""

import os
import sys
from datetime import datetime, timedelta
import time

# Додаємо проект в path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import bot_instance
from config import get_api_credentials

def debug_api_response():
    """Отримує одну закриту угоду і показує всі поля"""
    
    print("🔍 DEBUG: Отримання даних від Bybit API")
    print("=" * 70)
    
    try:
        # Отримуємо угоди за останній день
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (1 * 86400000)  # 1 день назад
        
        print(f"📅 Період: від {datetime.fromtimestamp(start_ms/1000)} до {datetime.now()}")
        print()
        
        r = bot_instance.session.get_closed_pnl(
            category="linear",
            startTime=int(start_ms),
            endTime=int(now_ms),
            limit=10
        )
        
        if r.get('retCode') != 0:
            print(f"❌ Ошибка API: {r}")
            return
        
        trades = r.get('result', {}).get('list', [])
        
        if not trades:
            print("⚠️ Нет закрытых угод за последний день")
            return
        
        print(f"✅ Найдено {len(trades)} угод\n")
        
        # Показуємо ПЕРШУ угоду детально
        t = trades[0]
        
        print("📊 ПЕРША УГОДА - ВСІ ПОЛЯ:")
        print("-" * 70)
        
        for key, value in sorted(t.items()):
            print(f"  {key:30s} = {value}")
        
        print("\n" + "=" * 70)
        print("🔍 РОЗМІРИ КОМІСІЙ:")
        print("-" * 70)
        
        # Показуємо комісії для всіх угод
        for i, t in enumerate(trades[:5], 1):
            symbol = t.get('symbol', 'N/A')
            
            # Перевіримо різні можливі назви полів
            possible_fee_keys = [
                'openingFee', 'closingFee', 'fundingFee',
                'takerFee', 'makerFee', 'totalFee',
                'feeRate', 'execFee',
                'tradeFee', 'commission'
            ]
            
            print(f"\n{i}. {symbol}:")
            
            has_fees = False
            for key in possible_fee_keys:
                if key in t:
                    value = t[key]
                    print(f"   ✓ {key}: {value} (тип: {type(value).__name__})")
                    has_fees = True
            
            if not has_fees:
                print(f"   ❌ Комісій не знайдено")
        
        print("\n" + "=" * 70)
        print("💡 РЕКОМЕНДАЦІЇ:")
        print("-" * 70)
        
        # Аналізуємо що знайшли
        if trades:
            t = trades[0]
            
            if 'openingFee' in t and t['openingFee'] != 0:
                print("✅ openingFee - ЗНАЙДЕНО и не нулевое")
            elif 'openingFee' in t:
                print("⚠️  openingFee - ЗНАЙДЕНО но все нули (может быть market maker)")
            else:
                print("❌ openingFee - НЕ ЗНАЙДЕНО")
            
            # Показуємо P&L
            pnl = t.get('closedPnl')
            print(f"\nP&L угоди: {pnl}")
            
            if pnl and float(pnl) != 0:
                print("✅ P&L ненулевой - значит API работает")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n🚀 SVV BOT - DEBUG СКРИПТ")
    print("=" * 70)
    debug_api_response()
    print("\n✅ Debug закончен")