"""
Migration Script - Перехід зі старої системи на нову
Використовується один раз для імпорту історичних даних
"""

import sys
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from config import get_api_credentials
from statistics_service import stats_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigration:
    """Міграція даних зі старої системи"""
    
    def __init__(self):
        try:
            api_key, api_secret = get_api_credentials()
            self.session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
            logger.info("✅ Connected to Bybit API")
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise
    
    def migrate_trades(self, days=30):
        """
        Імпортувати всі угоди за період
        """
        logger.info(f"🔄 Starting migration of trades for last {days} days...")
        
        try:
            now = datetime.now()
            all_trades = []
            imported_count = 0
            skipped_count = 0
            
            # Розбити на тижневі чанки для надійності
            for i in range(0, days, 7):
                chunk_days = min(7, days - i)
                end_dt = now - timedelta(days=i)
                start_dt = end_dt - timedelta(days=chunk_days)
                
                ts_end = int(end_dt.timestamp() * 1000)
                ts_start = int(start_dt.timestamp() * 1000)
                
                logger.info(f"📥 Fetching trades from {start_dt.date()} to {end_dt.date()}...")
                
                resp = self.session.get_closed_pnl(
                    category="linear",
                    startTime=ts_start,
                    endTime=ts_end,
                    limit=100
                )
                
                if resp['retCode'] == 0:
                    chunk_trades = resp['result']['list']
                    all_trades.extend(chunk_trades)
                    logger.info(f"   ✅ Found {len(chunk_trades)} trades")
                
                import time
                time.sleep(0.2)  # Rate limiting
            
            logger.info(f"📊 Total trades found: {len(all_trades)}")
            
            # Імпортувати в БД
            for trade in all_trades:
                try:
                    # Конвертувати сторону (Bybit API інвертує)
                    api_side = trade['side']
                    real_side = "Long" if api_side == "Sell" else "Short"
                    
                    entry_time = datetime.fromtimestamp(int(trade['createdTime']) / 1000)
                    exit_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000)
                    duration = (exit_time - entry_time).total_seconds() / 60
                    
                    price = float(trade['avgExitPrice'])
                    qty = float(trade['qty'])
                    volume = price * qty
                    pnl = float(trade['closedPnl'])
                    
                    trade_data = {
                        'order_id': trade['orderId'],
                        'symbol': trade['symbol'],
                        'side': real_side,
                        'qty': qty,
                        'entry_price': float(trade['avgEntryPrice']),
                        'exit_price': price,
                        'pnl': pnl,
                        'pnl_percent': (pnl / volume * 100) if volume > 0 else 0,
                        'volume_usd': volume,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'duration_minutes': int(duration)
                    }
                    
                    # Спроба зберегти (пропустить дублікати)
                    if stats_service.save_trade(trade_data):
                        imported_count += 1
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing trade: {e}")
                    continue
            
            logger.info(f"""
            ╔════════════════════════════════════════╗
            ║      MIGRATION COMPLETED               ║
            ╠════════════════════════════════════════╣
            ║  Total Found:    {len(all_trades):>6}              ║
            ║  Imported:       {imported_count:>6}              ║
            ║  Skipped:        {skipped_count:>6}              ║
            ╚════════════════════════════════════════╝
            """)
            
            return {
                'total': len(all_trades),
                'imported': imported_count,
                'skipped': skipped_count
            }
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return None
    
    def verify_migration(self):
        """
        Перевірити успішність міграції
        """
        logger.info("🔍 Verifying migration...")
        
        try:
            trades = stats_service.get_trades(days=30)
            
            if not trades:
                logger.warning("⚠️ No trades found in database!")
                return False
            
            # Статистика
            total_pnl = sum(t['pnl'] for t in trades)
            winning = sum(1 for t in trades if t['is_win'])
            losing = sum(1 for t in trades if not t['is_win'])
            win_rate = (winning / len(trades) * 100) if trades else 0
            
            logger.info(f"""
            ╔════════════════════════════════════════╗
            ║      VERIFICATION RESULTS              ║
            ╠════════════════════════════════════════╣
            ║  Total Trades:   {len(trades):>6}              ║
            ║  Win Rate:       {win_rate:>6.1f}%            ║
            ║  Total P&L:      ${total_pnl:>6.2f}         ║
            ║  Winners:        {winning:>6}              ║
            ║  Losers:         {losing:>6}              ║
            ╚════════════════════════════════════════╝
            """)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def cleanup_duplicates(self):
        """
        Видалити можливі дублікати (за order_id)
        """
        logger.info("🧹 Cleaning up duplicates...")
        
        from models import db_manager, Trade
        session = db_manager.get_session()
        
        try:
            # Знайти дублікати
            duplicates = session.query(Trade.order_id).group_by(Trade.order_id).having(
                func.count(Trade.order_id) > 1
            ).all()
            
            deleted_count = 0
            
            for (order_id,) in duplicates:
                # Залишити тільки перший запис
                trades_with_id = session.query(Trade).filter_by(order_id=order_id).all()
                
                if len(trades_with_id) > 1:
                    # Видалити всі крім першого
                    for trade in trades_with_id[1:]:
                        session.delete(trade)
                        deleted_count += 1
            
            session.commit()
            logger.info(f"✅ Deleted {deleted_count} duplicate records")
            
            return deleted_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Cleanup failed: {e}")
            return 0
        finally:
            session.close()

def main():
    """
    Головна функція міграції
    """
    print("""
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║       BYBIT BOT DATA MIGRATION TOOL           ║
    ║                                               ║
    ║  This script will import your trade history  ║
    ║  from Bybit API into the new database        ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Запитати підтвердження
    days = input("How many days of history to import? (default 30): ").strip()
    if not days:
        days = 30
    else:
        try:
            days = int(days)
        except:
            print("Invalid number, using 30 days")
            days = 30
    
    print(f"\n📊 Will import last {days} days of trade history")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Migration cancelled")
        return
    
    # Виконати міграцію
    migration = DataMigration()
    
    print("\n" + "="*50)
    print("STEP 1: Importing trades from Bybit API")
    print("="*50 + "\n")
    
    result = migration.migrate_trades(days=days)
    
    if not result:
        print("❌ Migration failed!")
        return
    
    print("\n" + "="*50)
    print("STEP 2: Verifying data integrity")
    print("="*50 + "\n")
    
    if not migration.verify_migration():
        print("⚠️ Verification found issues!")
        return
    
    print("\n" + "="*50)
    print("STEP 3: Cleaning up duplicates")
    print("="*50 + "\n")
    
    deleted = migration.cleanup_duplicates()
    
    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║         ✅ MIGRATION SUCCESSFUL! ✅           ║
    ║                                               ║
    ║  Your historical data has been imported      ║
    ║  You can now use the new system              ║
    ║                                               ║
    ║  Access your dashboard at:                   ║
    ║  http://localhost:10000/scanner              ║
    ║  http://localhost:10000/report               ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)
