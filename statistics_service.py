"""
Statistics Service - Зберігає історію, не видаляє
"""
from models import db_manager, WhaleSignal, CoinStatistics, Trade, CoinPerformance
from datetime import datetime, timedelta
from sqlalchemy import desc
import logging

logger = logging.getLogger(__name__)

class StatisticsService:
    def __init__(self):
        self.db = db_manager
    
    # Метод save_whale_signal залишаємо без змін (але без логів, як домовлялися)
    def save_whale_signal(self, signal_data):
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
            return True
        except: return False
        finally: session.close()

    def get_whale_signals(self, hours=24):
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            signals = session.query(WhaleSignal).filter(WhaleSignal.timestamp >= cutoff).order_by(desc(WhaleSignal.timestamp)).all()
            return [{
                'symbol': s.symbol, 'price': s.price, 'vol_inflow': s.volume_inflow,
                'spike_factor': s.spike_factor, 'price_change_interval': s.price_change_1min,
                'time': s.timestamp.strftime('%H:%M:%S')
            } for s in signals]
        finally: session.close()

    # 🔥 ОНОВЛЕНИЙ МЕТОД ЗБЕРЕЖЕННЯ УГОДИ
    def save_trade(self, trade_data):
        session = self.db.get_session()
        try:
            # Якщо такий order_id вже є - пропускаємо (щоб не дублювати при синхронізації)
            if trade_data.get('order_id'):
                existing = session.query(Trade).filter_by(order_id=trade_data['order_id']).first()
                if existing: return False
            
            trade = Trade(
                order_id=trade_data.get('order_id', f"MANUAL_{datetime.utcnow().timestamp()}"),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                qty=trade_data.get('qty', 0),
                entry_price=trade_data.get('entry_price', 0),
                exit_price=trade_data.get('exit_price', 0),
                pnl=trade_data.get('pnl', 0),
                is_win=trade_data.get('pnl', 0) > 0,
                exit_time=trade_data.get('exit_time', datetime.utcnow()),
                # Нові поля
                exit_reason=trade_data.get('exit_reason', 'Unknown'),
                exit_rsi=trade_data.get('exit_rsi', 0),
                exit_pressure=trade_data.get('exit_pressure', 0)
            )
            session.add(trade)
            session.commit()
            return True
        except Exception as e: 
            logger.error(f"Save trade error: {e}")
            return False
        finally: session.close()

    def get_trades(self, days=7):
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(Trade).filter(Trade.exit_time >= cutoff).order_by(desc(Trade.exit_time)).all()
            return [{
                'symbol': t.symbol, 'side': t.side, 'qty': t.qty, 'entry_price': t.entry_price,
                'exit_price': t.exit_price, 'pnl': round(t.pnl, 2), 'is_win': t.is_win,
                'exit_reason': t.exit_reason, 'exit_rsi': t.exit_rsi,
                'exit_time': t.exit_time.strftime('%d.%m %H:%M') if t.exit_time else None
            } for t in trades]
        finally: session.close()

    def cleanup_old_data(self, days=30):
        # Чистимо тільки старі сигнали сканера, АЛЕ НЕ УГОДИ
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            session.query(WhaleSignal).filter(WhaleSignal.timestamp < cutoff).delete()
            session.commit()
        except: pass
        finally: session.close()

stats_service = StatisticsService()
