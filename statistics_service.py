"""
Statistics Service - Оптимізований для чистої бази та логів
"""

from models import db_manager, WhaleSignal, CoinStatistics, Trade, CoinPerformance
from datetime import datetime, timedelta
from sqlalchemy import func, desc
import logging

logger = logging.getLogger(__name__)

class StatisticsService:
    """Сервіс для роботи зі статистикою та базою даних"""
    
    def __init__(self):
        self.db = db_manager
    
    # === 🔥 НОВИЙ МЕТОД: ОЧИЩЕННЯ ІСТОРІЇ МОНЕТИ ===
    def delete_coin_history(self, symbol):
        """Видаляє всю історію сигналів по конкретній монеті після закриття угоди"""
        session = self.db.get_session()
        try:
            # Видаляємо записи з таблиці сигналів
            deleted = session.query(WhaleSignal).filter_by(symbol=symbol).delete()
            session.commit()
            logger.info(f"🧹 Database cleaned for {symbol}: deleted {deleted} old signals.")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error cleaning coin history: {e}")
        finally:
            session.close()

    # === ЗБЕРЕЖЕННЯ СИГНАЛІВ (БЕЗ СПАМУ В ЛОГАХ) ===
    def save_whale_signal(self, signal_data):
        """Зберегти сигнал від сканера"""
        session = self.db.get_session()
        try:
            signal = WhaleSignal(
                symbol=signal_data['symbol'],
                timestamp=signal_data.get('timestamp_dt', datetime.utcnow()),
                price=signal_data['price'],
                volume_inflow=signal_data['vol_inflow'],
                spike_factor=signal_data['spike_factor'],
                price_change_1min=signal_data['price_change_interval'],
                turnover_24h=signal_data.get('turnover_24h', 0)
            )
            session.add(signal)
            session.commit()
            
            # Оновлюємо загальну статистику (опціонально)
            self._update_coin_statistics(session, signal_data['symbol'])
            
            # 🔇 ЛОГ ВИМКНЕНО, ЩОБ НЕ ЗАБИВАТИ КОНСОЛЬ
            # logger.info(f"✅ Saved whale signal: {signal_data['symbol']}")
            
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving whale signal: {e}")
            return False
        finally:
            session.close()

    # === ОТРИМАННЯ СИГНАЛІВ ===
    def get_whale_signals(self, hours=24):
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            signals = session.query(WhaleSignal).filter(WhaleSignal.timestamp >= cutoff).order_by(desc(WhaleSignal.timestamp)).all()
            return [{
                'symbol': s.symbol,
                'price': s.price,
                'vol_inflow': s.volume_inflow,
                'spike_factor': s.spike_factor,
                'price_change_interval': s.price_change_1min,
                'time': s.timestamp.strftime('%H:%M:%S')
            } for s in signals]
        finally:
            session.close()

    # === РОБОТА З УГОДАМИ ===
    def save_trade(self, trade_data):
        session = self.db.get_session()
        try:
            existing = session.query(Trade).filter_by(order_id=trade_data['order_id']).first()
            if existing: return False
            
            trade = Trade(
                order_id=trade_data['order_id'],
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                qty=trade_data['qty'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                pnl=trade_data['pnl'],
                exit_time=trade_data.get('exit_time'),
                is_win=trade_data['pnl'] > 0
            )
            session.add(trade)
            session.commit()
            return True
        except: return False
        finally: session.close()

    def get_trades(self, days=7):
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(Trade).filter(Trade.exit_time >= cutoff).order_by(desc(Trade.exit_time)).all()
            return [{
                'symbol': t.symbol,
                'side': t.side,
                'qty': t.qty,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'is_win': t.is_win,
                'exit_time': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else None
            } for t in trades]
        finally: session.close()

    # === ДОПОМІЖНІ ===
    def _update_coin_statistics(self, session, symbol):
        # Спрощена версія для економії ресурсів
        try:
            coin_stat = session.query(CoinStatistics).filter_by(symbol=symbol).first()
            if not coin_stat:
                coin_stat = CoinStatistics(symbol=symbol, first_seen=datetime.utcnow())
                session.add(coin_stat)
            coin_stat.last_updated = datetime.utcnow()
            session.commit()
        except: pass

    def cleanup_old_data(self, days=30):
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            session.query(WhaleSignal).filter(WhaleSignal.timestamp < cutoff).delete()
            session.commit()
        except: pass
        finally: session.close()

stats_service = StatisticsService()
