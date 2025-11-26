"""
Statistics Service - Updated for Monitor Logs
"""
from models import db_manager, WhaleSignal, Trade, TradeMonitorLog
from datetime import datetime, timedelta
from sqlalchemy import desc
import logging

logger = logging.getLogger(__name__)

class StatisticsService:
    def __init__(self):
        self.db = db_manager

    # === 🔥 НОВИЙ МЕТОД: ЗАПИС СТАТИСТИКИ АКТИВНОЇ УГОДИ ===
    def save_monitor_log(self, data):
        session = self.db.get_session()
        try:
            log = TradeMonitorLog(
                symbol=data['symbol'],
                timestamp=datetime.utcnow(),
                current_price=data['price'],
                current_pnl=data['pnl'],
                rsi=data['rsi'],
                pressure=data['pressure'],
                session_id=data.get('session_id', 'unknown')
            )
            session.add(log)
            session.commit()
        except: pass
        finally: session.close()

    # === 🔥 ОТРИМАННЯ ЛОГІВ ДЛЯ ЛІВОГО БЛОКУ ===
    def get_monitor_logs(self, limit=50):
        session = self.db.get_session()
        try:
            # Беремо останні записи
            logs = session.query(TradeMonitorLog).order_by(desc(TradeMonitorLog.timestamp)).limit(limit).all()
            return [{
                'time': l.timestamp.strftime('%H:%M:%S'),
                'symbol': l.symbol,
                'price': l.current_price,
                'pnl': l.current_pnl,
                'rsi': l.rsi,
                'pressure': l.pressure
            } for l in logs]
        finally: session.close()

    # ... (Інші методи save_trade, get_trades, save_whale_signal залишаємо без змін, як було) ...
    # Я скопіюю їх скорочено, щоб файл був повним:
    
    def save_whale_signal(self, d):
        s = self.db.get_session()
        try:
            w = WhaleSignal(symbol=d['symbol'], timestamp=d['timestamp_dt'], price=d['price'], volume_inflow=d['vol_inflow'], spike_factor=d['spike_factor'], price_change_1min=d['price_change_interval'], turnover_24h=d.get('turnover_24h',0))
            s.add(w); s.commit()
        except: pass
        finally: s.close()

    def get_whale_signals(self, hours=24):
        s = self.db.get_session()
        try:
            c = datetime.utcnow() - timedelta(hours=hours)
            res = s.query(WhaleSignal).filter(WhaleSignal.timestamp >= c).order_by(desc(WhaleSignal.timestamp)).all()
            return [{'symbol':x.symbol, 'price':x.price, 'vol_inflow':x.volume_inflow, 'spike_factor':x.spike_factor, 'price_change_interval':x.price_change_1min, 'time':x.timestamp.strftime('%H:%M:%S')} for x in res]
        finally: s.close()

    def save_trade(self, d):
        s = self.db.get_session()
        try:
            if d.get('order_id'):
                if s.query(Trade).filter_by(order_id=d['order_id']).first(): return
            t = Trade(order_id=d.get('order_id'), symbol=d['symbol'], side=d['side'], qty=d.get('qty',0), entry_price=d.get('entry_price',0), exit_price=d.get('exit_price',0), pnl=d.get('pnl',0), is_win=d.get('pnl',0)>0, exit_time=d.get('exit_time'), exit_reason=d.get('exit_reason'), exit_rsi=d.get('exit_rsi'), exit_pressure=d.get('exit_pressure'))
            s.add(t); s.commit()
        except: pass
        finally: s.close()

    def get_trades(self, days=7):
        s = self.db.get_session()
        try:
            c = datetime.utcnow() - timedelta(days=days)
            res = s.query(Trade).filter(Trade.exit_time >= c).order_by(desc(Trade.exit_time)).all()
            return [{'symbol':t.symbol, 'side':t.side, 'qty':t.qty, 'entry_price':t.entry_price, 'exit_price':t.exit_price, 'pnl':round(t.pnl,2), 'is_win':t.is_win, 'exit_reason':t.exit_reason, 'exit_rsi':t.exit_rsi, 'exit_time':t.exit_time.strftime('%d.%m %H:%M') if t.exit_time else None} for t in res]
        finally: s.close()
        
    def cleanup_old_data(self, days=7):
        s = self.db.get_session()
        try:
            c = datetime.utcnow() - timedelta(days=days)
            s.query(WhaleSignal).filter(WhaleSignal.timestamp < c).delete()
            s.query(TradeMonitorLog).filter(TradeMonitorLog.timestamp < c).delete() # Чистимо логи моніторингу теж
            s.commit()
        except: pass
        finally: s.close()

stats_service = StatisticsService()
