"""
Analytics Utilities - Розширена аналітика та звіти
"""

from models import db_manager, WhaleSignal, Trade, CoinStatistics, CoinPerformance
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_
import json

class AdvancedAnalytics:
    """Розширена аналітика для глибокого аналізу даних"""
    
    def __init__(self):
        self.db = db_manager
    
    def get_market_momentum(self, hours=24):
        """
        Визначити загальний імпульс ринку на основі whale активності
        Returns: score від -100 до +100
        """
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            signals = session.query(WhaleSignal).filter(
                WhaleSignal.timestamp >= cutoff
            ).all()
            
            if not signals:
                return 0
            
            # Розрахувати скор на основі price changes та inflows
            bullish_score = 0
            bearish_score = 0
            
            for signal in signals:
                weight = signal.volume_inflow / 1000000  # Вага в мільйонах
                
                if signal.price_change_1min > 0:
                    bullish_score += signal.price_change_1min * weight
                else:
                    bearish_score += abs(signal.price_change_1min) * weight
            
            total_score = bullish_score + bearish_score
            if total_score == 0:
                return 0
            
            # Нормалізувати до -100...+100
            momentum = ((bullish_score - bearish_score) / total_score) * 100
            return round(momentum, 2)
            
        finally:
            session.close()
    
    def get_coin_risk_score(self, symbol):
        """
        Розрахувати risk score для монети (0-100)
        Чим нижче - тим безпечніше
        """
        session = self.db.get_session()
        try:
            # Отримати статистику
            coin_stat = session.query(CoinStatistics).filter_by(symbol=symbol).first()
            coin_perf = session.query(CoinPerformance).filter_by(symbol=symbol).first()
            
            if not coin_stat:
                return None
            
            risk_score = 50  # Baseline
            
            # Фактор 1: Win rate (якщо є історія торгівлі)
            if coin_perf and coin_perf.total_trades >= 5:
                if coin_perf.win_rate < 40:
                    risk_score += 20
                elif coin_perf.win_rate > 60:
                    risk_score -= 15
            
            # Фактор 2: Волатильність (на основі spike factor)
            if coin_stat.max_spike_factor > 5:
                risk_score += 15
            elif coin_stat.max_spike_factor < 2:
                risk_score -= 10
            
            # Фактор 3: Співвідношення позитивних/негативних сигналів
            if coin_stat.total_signals > 0:
                neg_ratio = coin_stat.negative_signals / coin_stat.total_signals
                if neg_ratio > 0.6:
                    risk_score += 10
                elif neg_ratio < 0.4:
                    risk_score -= 10
            
            # Обмежити 0-100
            risk_score = max(0, min(100, risk_score))
            
            return round(risk_score, 1)
            
        finally:
            session.close()
    
    def get_best_trading_hours(self, days=7):
        """
        Визначити найкращі години для торгівлі на основі історії
        Returns: dict з годинами та їх win rate
        """
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            trades = session.query(Trade).filter(
                Trade.exit_time >= cutoff
            ).all()
            
            if not trades:
                return {}
            
            # Групувати по годинах
            hours_stats = {}
            
            for trade in trades:
                if not trade.exit_time:
                    continue
                
                hour = trade.exit_time.hour
                
                if hour not in hours_stats:
                    hours_stats[hour] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                
                if trade.is_win:
                    hours_stats[hour]['wins'] += 1
                else:
                    hours_stats[hour]['losses'] += 1
                
                hours_stats[hour]['total_pnl'] += trade.pnl
            
            # Розрахувати win rate для кожної години
            result = {}
            for hour, stats in hours_stats.items():
                total = stats['wins'] + stats['losses']
                win_rate = (stats['wins'] / total * 100) if total > 0 else 0
                
                result[hour] = {
                    'win_rate': round(win_rate, 1),
                    'total_trades': total,
                    'avg_pnl': round(stats['total_pnl'] / total, 2) if total > 0 else 0
                }
            
            # Сортувати по win rate
            sorted_hours = sorted(result.items(), key=lambda x: x[1]['win_rate'], reverse=True)
            return dict(sorted_hours)
            
        finally:
            session.close()
    
    def get_coin_correlation_matrix(self, symbols, days=7):
        """
        Матриця кореляції між монетами на основі whale сигналів
        """
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Отримати сигнали для всіх монет
            coin_signals = {}
            
            for symbol in symbols:
                signals = session.query(WhaleSignal).filter(
                    WhaleSignal.symbol == symbol,
                    WhaleSignal.timestamp >= cutoff
                ).order_by(WhaleSignal.timestamp).all()
                
                # Створити часовий ряд змін цін
                coin_signals[symbol] = [s.price_change_1min for s in signals]
            
            # Простий розрахунок кореляції (Pearson)
            matrix = {}
            
            for sym1 in symbols:
                matrix[sym1] = {}
                for sym2 in symbols:
                    if sym1 == sym2:
                        matrix[sym1][sym2] = 1.0
                    else:
                        # Спрощена кореляція (для більш точної використати numpy)
                        corr = self._simple_correlation(
                            coin_signals.get(sym1, []),
                            coin_signals.get(sym2, [])
                        )
                        matrix[sym1][sym2] = corr
            
            return matrix
            
        finally:
            session.close()
    
    def _simple_correlation(self, x, y):
        """Спрощений розрахунок кореляції"""
        if not x or not y or len(x) < 2 or len(y) < 2:
            return 0
        
        # Вирівняти довжини
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Середні значення
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        # Коваріація та дисперсії
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        variance_x = sum((xi - mean_x) ** 2 for xi in x)
        variance_y = sum((yi - mean_y) ** 2 for yi in y)
        
        if variance_x == 0 or variance_y == 0:
            return 0
        
        correlation = covariance / (variance_x * variance_y) ** 0.5
        return round(correlation, 3)
    
    def generate_daily_report(self, date=None):
        """
        Згенерувати денний звіт з усією статистикою
        """
        if date is None:
            date = datetime.utcnow().date()
        
        session = self.db.get_session()
        try:
            start_dt = datetime.combine(date, datetime.min.time())
            end_dt = datetime.combine(date, datetime.max.time())
            
            # Whale signals за день
            signals = session.query(WhaleSignal).filter(
                WhaleSignal.timestamp >= start_dt,
                WhaleSignal.timestamp <= end_dt
            ).all()
            
            # Trades за день
            trades = session.query(Trade).filter(
                Trade.exit_time >= start_dt,
                Trade.exit_time <= end_dt
            ).all()
            
            # Статистика
            report = {
                'date': date.strftime('%Y-%m-%d'),
                'whale_activity': {
                    'total_signals': len(signals),
                    'total_inflow': sum(s.volume_inflow for s in signals),
                    'unique_coins': len(set(s.symbol for s in signals)),
                    'avg_spike_factor': sum(s.spike_factor for s in signals) / len(signals) if signals else 0
                },
                'trading': {
                    'total_trades': len(trades),
                    'winning_trades': sum(1 for t in trades if t.is_win),
                    'total_pnl': sum(t.pnl for t in trades),
                    'win_rate': (sum(1 for t in trades if t.is_win) / len(trades) * 100) if trades else 0
                },
                'top_coins_by_inflow': self._get_top_coins_day(session, start_dt, end_dt),
                'market_momentum': self.get_market_momentum(hours=24)
            }
            
            return report
            
        finally:
            session.close()
    
    def _get_top_coins_day(self, session, start_dt, end_dt, limit=5):
        """Топ монет за день по вливанням"""
        query = session.query(
            WhaleSignal.symbol,
            func.sum(WhaleSignal.volume_inflow).label('total_inflow'),
            func.count(WhaleSignal.id).label('signal_count')
        ).filter(
            WhaleSignal.timestamp >= start_dt,
            WhaleSignal.timestamp <= end_dt
        ).group_by(
            WhaleSignal.symbol
        ).order_by(
            desc('total_inflow')
        ).limit(limit).all()
        
        return [{
            'symbol': row.symbol,
            'total_inflow': round(row.total_inflow, 0),
            'signals': row.signal_count
        } for row in query]
    
    def export_to_json(self, output_file='analytics_export.json'):
        """
        Експортувати всю аналітику в JSON файл
        """
        data = {
            'export_date': datetime.utcnow().isoformat(),
            'market_momentum_24h': self.get_market_momentum(hours=24),
            'best_trading_hours': self.get_best_trading_hours(days=7),
            'daily_report': self.generate_daily_report()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file

# Глобальний інстанс
analytics = AdvancedAnalytics()


# === HELPER FUNCTIONS ===

def get_trade_recommendations(limit=10):
    """
    Отримати рекомендації на основі історичної продуктивності
    """
    session = db_manager.get_session()
    try:
        # Знайти монети з хорошим win rate та позитивним P&L
        perfs = session.query(CoinPerformance).filter(
            CoinPerformance.total_trades >= 5,
            CoinPerformance.win_rate >= 55,
            CoinPerformance.total_pnl > 0
        ).order_by(
            desc(CoinPerformance.win_rate)
        ).limit(limit).all()
        
        recommendations = []
        
        for perf in perfs:
            # Розрахувати risk score
            risk_score = analytics.get_coin_risk_score(perf.symbol)
            
            recommendations.append({
                'symbol': perf.symbol,
                'win_rate': round(perf.win_rate, 1),
                'total_pnl': round(perf.total_pnl, 2),
                'total_trades': perf.total_trades,
                'risk_score': risk_score,
                'recommendation': 'LOW RISK' if risk_score < 40 else 'MEDIUM RISK' if risk_score < 60 else 'HIGH RISK'
            })
        
        return recommendations
        
    finally:
        session.close()


def get_performance_summary():
    """
    Швидкий summary всієї продуктивності
    """
    session = db_manager.get_session()
    try:
        total_pnl = session.query(func.sum(Trade.pnl)).scalar() or 0
        total_trades = session.query(func.count(Trade.id)).scalar() or 0
        winning_trades = session.query(func.count(Trade.id)).filter(Trade.is_win == True).scalar() or 0
        
        best_coin = session.query(
            CoinPerformance.symbol,
            CoinPerformance.total_pnl
        ).order_by(
            desc(CoinPerformance.total_pnl)
        ).first()
        
        worst_coin = session.query(
            CoinPerformance.symbol,
            CoinPerformance.total_pnl
        ).order_by(
            CoinPerformance.total_pnl
        ).first()
        
        return {
            'total_pnl': round(total_pnl, 2),
            'total_trades': total_trades,
            'win_rate': round((winning_trades / total_trades * 100), 1) if total_trades > 0 else 0,
            'best_coin': best_coin.symbol if best_coin else None,
            'best_coin_pnl': round(best_coin.total_pnl, 2) if best_coin else 0,
            'worst_coin': worst_coin.symbol if worst_coin else None,
            'worst_coin_pnl': round(worst_coin.total_pnl, 2) if worst_coin else 0,
            'market_momentum': analytics.get_market_momentum(hours=24)
        }
        
    finally:
        session.close()
