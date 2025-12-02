"""
Smart Money Tracker
Monitoring module that watches accumulated Order Blocks in DB
and executes trades when price re-visits the zone.
"""
import time
import logging
import threading
from datetime import datetime
from models import db_manager, OrderBlock
from bot import bot_instance
from settings_manager import settings

logger = logging.getLogger(__name__)

class SmartMoneyTracker:
    def __init__(self):
        self.running = True
        self.interval = 10 # Перевіряти кожні 10 сек
        
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🦅 Smart Money Tracker Started")

    def loop(self):
        while self.running:
            try:
                self.check_pending_blocks()
            except Exception as e:
                logger.error(f"Tracker Loop Error: {e}")
            time.sleep(self.interval)

    def check_pending_blocks(self):
        session = db_manager.get_session()
        try:
            # 1. Отримуємо всі PENDING блоки
            pending_obs = session.query(OrderBlock).filter_by(status='PENDING').all()
            
            if not pending_obs:
                return

            for ob in pending_obs:
                symbol = ob.symbol
                
                # 2. Отримуємо актуальну ціну
                current_price = bot_instance.get_price(symbol)
                if current_price == 0: continue

                # 3. Перевірка умов входу (RETEST LOGIC)
                should_execute = False
                
                # Для BUY: Ціна повинна бути в зоні або нижче entry, але вище SL
                if ob.ob_type == "Buy":
                    # Логіка: Якщо ми зловили блок, коли ціна була вище, чекаємо поки спуститься (ретест)
                    # Або входимо відразу, якщо ми вже в зоні.
                    # Спрощена перевірка: чи ціна в межах [SL...Entry * 1.002]
                    if ob.sl_price < current_price <= (ob.entry_price * 1.005):
                        should_execute = True
                        
                # Для SELL: Ціна повинна бути в зоні або вище entry, але нижче SL
                elif ob.ob_type == "Sell":
                    if ob.sl_price > current_price >= (ob.entry_price * 0.995):
                        should_execute = True

                # 4. Виконання
                if should_execute:
                    logger.info(f"🦅 TRACKER TRIGGER: {symbol} {ob.ob_type} at {current_price}")
                    
                    # Відправка ордеру
                    result = bot_instance.place_order({
                        "symbol": symbol,
                        "action": ob.ob_type, # "Buy" or "Sell"
                        "sl_price": ob.sl_price
                    })
                    
                    if result.get("status") == "ok":
                        ob.status = 'FILLED'
                        logger.info(f"✅ Order Filled for {symbol}")
                    else:
                        logger.warning(f"❌ Order Failed: {result}")
                        # Можливо, варто спробувати пізніше або помітити як INVALID
                
                # 5. Інвалідація (якщо ціна вибила SL ще до входу)
                if ob.ob_type == "Buy" and current_price < ob.sl_price:
                    ob.status = 'INVALID'
                elif ob.ob_type == "Sell" and current_price > ob.sl_price:
                    ob.status = 'INVALID'
                    
            session.commit()
            
        except Exception as e:
            logger.error(f"Tracker logic error: {e}")
        finally:
            session.close()

# Глобальний екземпляр
tracker = SmartMoneyTracker()