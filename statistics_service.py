"""
Statistics Service - Професійна аналітика та агрегація даних
"""

from models import db_manager, WhaleSignal, CoinStatistics, Trade, TradingSession, CoinPerformance
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_
import logging

logger = logging.getLogger(__name__)

class StatisticsService:
    """Сервіс для роботи зі статистикою"""
    
    def __init__(self):
        self.db = db_manager
    
    # === WHALE SIGNALS ===
    
    def save_whale_signal(self, signal_data):
        """Зберегти сигнал від whale scanner"""
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
            
            # Оновити статистику монети
            self._update_coin_statistics(session, signal_data['symbol'])
            
            logger.info(f"✅ Saved whale signal: {signal_data['symbol']}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving whale signal: {e}")
            return False
        finally:
            session.close()
    
    def get_whale_signals(self, hours=24, symbol=None, min_inflow=None):
        """Отримати сигнали за період"""
        session = self.db.get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = session.query(WhaleSignal).filter(WhaleSignal.timestamp >= cutoff_time)
            
            if symbol:
                query = query.filter(WhaleSignal.symbol == symbol)
            
            if min_inflow:
                query = query.filter(WhaleSignal.volume_inflow >= min_inflow)
            
            signals = query.order_by(desc(WhaleSignal.timestamp)).all()
            return [self._signal_to_dict(s) for s in signals]
        finally:
            session.close()
    
    def _signal_to_dict(self, signal):
        """Конвертувати сигнал в dict"""
        return {
            'symbol': signal.symbol,
            'timestamp': signal.timestamp,
            'price': signal.price,
            'vol_inflow': signal.volume_inflow,
            'spike_factor': signal.spike_factor,
            'price_change_interval': signal.price_change_1min,
            'time': signal.timestamp.strftime('%H:%M:%S')
        }
    
    # === COIN STATISTICS ===
    
    def _update_coin_statistics(self, session, symbol):
        """Оновити агреговану статистику монети"""
        try:
            # Отримати або створити запис
            coin_stat = session.query(CoinStatistics).filter_by(symbol=symbol).first()
            if not coin_stat:
                coin_stat = CoinStatistics(symbol=symbol, first_seen=datetime.utcnow())
                session.add(coin_stat)
            
            # Розрахувати статистику за періоди
            now = datetime.utcnow()
            
            # 24 години
            signals_24h = session.query(WhaleSignal).filter(
                WhaleSignal.symbol == symbol,
                WhaleSignal.timestamp >= now - timedelta(hours=24)
            ).all()
            
            # 7 днів
            signals_7d = session.query(WhaleSignal).filter(
                WhaleSignal.symbol == symbol,
                WhaleSignal.timestamp >= now - timedelta(days=7)
            ).all()
            
            # 30 днів
            signals_30d = session.query(WhaleSignal).filter(
                WhaleSignal.symbol == symbol,
                WhaleSignal.timestamp >= now - timedelta(days=30)
            ).all()
            
            # Оновити дані
            coin_stat.total_signals = len(signals_30d)
            coin_stat.total_inflow_24h = sum(s.volume_inflow for s in signals_24h)
            coin_stat.total_inflow_7d = sum(s.volume_inflow for s in signals_7d)
            coin_stat.total_inflow_30d = sum(s.volume_inflow for s in signals_30d)
            
            if signals_30d:
                coin_stat.avg_spike_factor = sum(s.spike_factor for s in signals_30d) / len(signals_30d)
                coin_stat.max_spike_factor = max(s.spike_factor for s in signals_30d)
                coin_stat.avg_price_change = sum(s.price_change_1min for s in signals_30d) / len(signals_30d)
                
                coin_stat.positive_signals = sum(1 for s in signals_30d if s.price_change_1min > 0)
                coin_stat.negative_signals = sum(1 for s in signals_30d if s.price_change_1min < 0)
            
            coin_stat.last_seen = now
            coin_stat.last_updated = now
            
            session.commit()
        except Exception as e:
            logger.error(f"Error updating coin statistics: {e}")
    
    def get_top_coins(self, period_hours=24, limit=20, sort_by='inflow'):
        """Отримати топ монет за період"""
        session = self.db.get_session()
        try:
            query = session.query(CoinStatistics)
            
            # Фільтр за оновленням
            cutoff = datetime.utcnow() - timedelta(hours=period_hours)
            query = query.filter(CoinStatistics.last_updated >= cutoff)
            
            # Сортування
            if sort_by == 'inflow':
                query = query.order_by(desc(CoinStatistics.total_inflow_24h))
            elif sort_by == 'spike':
                query = query.order_by(desc(CoinStatistics.max_spike_factor))
            elif sort_by == 'signals':
                query = query.order_by(desc(CoinStatistics.total_signals))
            
            coins = query.limit(limit).all()
            return [self._coin_stat_to_dict(c) for c in coins]
        finally:
            session.close()
    
    def _coin_stat_to_dict(self, coin_stat):
        """Конвертувати статистику монети в dict"""
        return {
            'symbol': coin_stat.symbol,
            'total_signals': coin_stat.total_signals,
            'inflow_24h': coin_stat.total_inflow_24h,
            'inflow_7d': coin_stat.total_inflow_7d,
            'avg_spike': coin_stat.avg_spike_factor,
            'max_spike': coin_stat.max_spike_factor,
            'avg_change': coin_stat.avg_price_change,
            'positive_ratio': coin_stat.positive_signals / coin_stat.total_signals if coin_stat.total_signals > 0 else 0
        }
    
    # === TRADES ===
    
    def save_trade(self, trade_data):
        """Зберегти закриту угоду"""
        session = self.db.get_session()
        try:
            # Перевірити чи вже існує
            existing = session.query(Trade).filter_by(order_id=trade_data['order_id']).first()
            if existing:
                return False
            
            trade = Trade(
                order_id=trade_data['order_id'],
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                qty=trade_data['qty'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                pnl=trade_data['pnl'],
                pnl_percent=trade_data.get('pnl_percent'),
                is_win=trade_data['pnl'] > 0,
                leverage=trade_data.get('leverage'),
                volume_usd=trade_data.get('volume_usd'),
                entry_time=trade_data.get('entry_time'),
                exit_time=trade_data.get('exit_time'),
                duration_minutes=trade_data.get('duration_minutes')
            )
            session.add(trade)
            session.commit()
            
            # Оновити продуктивність монети
            self._update_coin_performance(session, trade_data['symbol'])
            
            logger.info(f"✅ Saved trade: {trade_data['symbol']} P&L: {trade_data['pnl']}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving trade: {e}")
            return False
        finally:
            session.close()
    
    def get_trades(self, days=7, symbol=None):
        """Отримати угоди за період"""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = session.query(Trade).filter(Trade.exit_time >= cutoff)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            trades = query.order_by(desc(Trade.exit_time)).all()
            return [self._trade_to_dict(t) for t in trades]
        finally:
            session.close()
    
    def _trade_to_dict(self, trade):
        """Конвертувати угоду в dict"""
        return {
            'symbol': trade.symbol,
            'side': trade.side,
            'qty': trade.qty,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'is_win': trade.is_win,
            'exit_time': trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else None
        }
    
    # === COIN PERFORMANCE ===
    
    def _update_coin_performance(self, session, symbol):
        """Оновити продуктивність монети"""
        try:
            perf = session.query(CoinPerformance).filter_by(symbol=symbol).first()
            if not perf:
                perf = CoinPerformance(symbol=symbol)
                session.add(perf)
            
            # Отримати всі угоди по монеті
            trades = session.query(Trade).filter(Trade.symbol == symbol).all()
            
            if not trades:
                return
            
            perf.total_trades = len(trades)
            perf.winning_trades = sum(1 for t in trades if t.is_win)
            perf.losing_trades = sum(1 for t in trades if not t.is_win)
            perf.win_rate = (perf.winning_trades / perf.total_trades * 100) if perf.total_trades > 0 else 0
            
            perf.total_pnl = sum(t.pnl for t in trades)
            perf.avg_pnl_per_trade = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
            perf.best_trade = max(t.pnl for t in trades)
            perf.worst_trade = min(t.pnl for t in trades)
            
            perf.total_volume = sum(t.volume_usd for t in trades if t.volume_usd)
            perf.avg_volume_per_trade = perf.total_volume / perf.total_trades if perf.total_trades > 0 else 0
            
            long_trades = [t for t in trades if t.side == 'Long']
            short_trades = [t for t in trades if t.side == 'Short']
            
            perf.long_trades = len(long_trades)
            perf.short_trades = len(short_trades)
            
            if long_trades:
                long_wins = sum(1 for t in long_trades if t.is_win)
                perf.long_win_rate = (long_wins / len(long_trades) * 100)
            
            if short_trades:
                short_wins = sum(1 for t in short_trades if t.is_win)
                perf.short_win_rate = (short_wins / len(short_trades) * 100)
            
            perf.last_updated = datetime.utcnow()
            session.commit()
        except Exception as e:
            logger.error(f"Error updating coin performance: {e}")
    
    def get_coin_performance(self, limit=50, min_trades=3):
        """Отримати продуктивність всіх монет"""
        session = self.db.get_session()
        try:
            perfs = session.query(CoinPerformance)\
                .filter(CoinPerformance.total_trades >= min_trades)\
                .order_by(desc(CoinPerformance.total_pnl))\
                .limit(limit)\
                .all()
            
            return [self._performance_to_dict(p) for p in perfs]
        finally:
            session.close()
    
    def _performance_to_dict(self, perf):
        """Конвертувати продуктивність в dict"""
        return {
            'symbol': perf.symbol,
            'total_trades': perf.total_trades,
            'win_rate': round(perf.win_rate, 1),
            'total_pnl': round(perf.total_pnl, 2),
            'avg_pnl': round(perf.avg_pnl_per_trade, 2),
            'best_trade': round(perf.best_trade, 2),
            'worst_trade': round(perf.worst_trade, 2),
            'long_wr': round(perf.long_win_rate, 1) if perf.long_win_rate else 0,
            'short_wr': round(perf.short_win_rate, 1) if perf.short_win_rate else 0
        }
    
    # === ANALYTICS ===
    
    def get_whale_heatmap(self, hours=24):
        """Тепловая карта активності китів по монетам та часу"""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            # Групувати по символу та годині
            query = session.query(
                WhaleSignal.symbol,
                func.strftime('%H', WhaleSignal.timestamp).label('hour'),
                func.count(WhaleSignal.id).label('signal_count'),
                func.sum(WhaleSignal.volume_inflow).label('total_inflow')
            ).filter(
                WhaleSignal.timestamp >= cutoff
            ).group_by(
                WhaleSignal.symbol,
                func.strftime('%H', WhaleSignal.timestamp)
            ).order_by(
                desc('total_inflow')
            ).all()
            
            return [{
                'symbol': row.symbol,
                'hour': row.hour,
                'signals': row.signal_count,
                'inflow': row.total_inflow
            } for row in query]
        finally:
            session.close()
    
    def get_correlation_data(self, symbol1, symbol2, days=7):
        """Кореляція між сигналами двох монет"""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            signals1 = session.query(WhaleSignal).filter(
                WhaleSignal.symbol == symbol1,
                WhaleSignal.timestamp >= cutoff
            ).order_by(WhaleSignal.timestamp).all()
            
            signals2 = session.query(WhaleSignal).filter(
                WhaleSignal.symbol == symbol2,
                WhaleSignal.timestamp >= cutoff
            ).order_by(WhaleSignal.timestamp).all()
            
            # Простий підрахунок співпадінь у часі (в межах 5 хв)
            correlations = []
            for s1 in signals1:
                for s2 in signals2:
                    time_diff = abs((s1.timestamp - s2.timestamp).total_seconds())
                    if time_diff <= 300:  # 5 хвилин
                        correlations.append({
                            'time': s1.timestamp,
                            'symbol1_change': s1.price_change_1min,
                            'symbol2_change': s2.price_change_1min,
                            'time_diff': time_diff
                        })
            
            return correlations
        finally:
            session.close()
    
    def cleanup_old_data(self, days=30):
        """Видалити старі дані"""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Видалити старі сигнали
            deleted_signals = session.query(WhaleSignal).filter(
                WhaleSignal.timestamp < cutoff
            ).delete()
            
            session.commit()
            logger.info(f"🗑️ Cleaned up {deleted_signals} old whale signals")
            return deleted_signals
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up data: {e}")
            return 0
        finally:
            session.close()

# Глобальний інстанс
stats_service = StatisticsService()
