from models import db_manager, Trade, TradeMonitorLog
from datetime import datetime, timedelta
from sqlalchemy import desc
class StatisticsService:
    def __init__(self): self.db = db_manager
    def save_trade(self, d):
        s = self.db.get_session()
        try:
            if d.get('order_id') and s.query(Trade).filter_by(order_id=d['order_id']).first(): return
            t = Trade(order_id=d.get('order_id'), symbol=d['symbol'], side=d['side'], qty=d.get('qty',0), entry_price=d.get('entry_price',0), exit_price=d.get('exit_price',0), pnl=d.get('pnl',0), is_win=d.get('pnl',0)>0, exit_time=d.get('exit_time'), exit_reason=d.get('exit_reason'))
            s.add(t); s.commit()
        except: pass
        finally: s.close()
    def save_monitor_log(self, data):
        s = self.db.get_session()
        try:
            l = TradeMonitorLog(symbol=data['symbol'], timestamp=datetime.utcnow(), current_price=data['price'], current_pnl=data['pnl'], rsi=data['rsi'], pressure=data['pressure'])
            s.add(l); s.commit()
        except: pass
        finally: s.close()
    def get_trades(self, days=7):
        s = self.db.get_session()
        try:
            d = datetime.utcnow() - timedelta(days=days)
            return [t.__dict__ for t in s.query(Trade).filter(Trade.exit_time >= d).order_by(desc(Trade.exit_time)).all()]
        finally: s.close()
stats_service = StatisticsService()
