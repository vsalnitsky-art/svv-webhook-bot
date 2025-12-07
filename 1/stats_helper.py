# ===== ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ПРОФЕСІЙНОГО ОГЛЯДУ РИНКУ =====
# Добавьте цей код у main_app.py перед route функціями

from collections import Counter
from statistics import mean

def calculate_stats(trades):
    """
    Розраховує детальну статистику торгів
    
    Повертає:
    {
        'total_trades': кількість угод,
        'win_trades': кількість прибилків,
        'loss_trades': кількість збитків,
        'win_rate': % успішних угод,
        'total_pnl': загальний P&L,
        'avg_pnl': середній P&L на угоду,
        'total_volume': загальний обсяг,
        'avg_trade_size': середній розмір угоди,
        'best_trade': найкраща угода,
        'worst_trade': найгірша угода,
        'consecutive_wins': найдовше вінінг стрик,
        'consecutive_losses': найдовше лозінг стрик,
        'contract_stats': статистика по контрактах (для рейтингу)
    }
    """
    if not trades:
        return {
            'total_trades': 0, 'win_trades': 0, 'loss_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            'total_volume': 0, 'avg_trade_size': 0,
            'best_trade': 0, 'worst_trade': 0,
            'consecutive_wins': 0, 'consecutive_losses': 0,
            'contract_stats': {}
        }
    
    # Базова статистика
    total = len(trades)
    wins = len([t for t in trades if t.get('pnl', 0) > 0])
    losses = total - wins
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = round((wins / total * 100) if total > 0 else 0, 1)
    avg_pnl = round(total_pnl / total if total > 0 else 0, 2)
    
    # Обсяги
    total_volume = sum(t.get('qty', 0) for t in trades)
    avg_trade_size = round(total_volume / total if total > 0 else 0, 2)
    
    # Кращі/гірші угоди
    best = max((t.get('pnl', 0) for t in trades), default=0)
    worst = min((t.get('pnl', 0) for t in trades), default=0)
    
    # Стреки
    wins_streak = 0
    curr_wins = 0
    for t in trades:
        if t.get('pnl', 0) > 0:
            curr_wins += 1
            wins_streak = max(wins_streak, curr_wins)
        else:
            curr_wins = 0
    
    losses_streak = 0
    curr_losses = 0
    for t in trades:
        if t.get('pnl', 0) <= 0:
            curr_losses += 1
            losses_streak = max(losses_streak, curr_losses)
        else:
            curr_losses = 0
    
    # Статистика по контрактах
    contract_stats = {}
    for t in trades:
        symbol = t.get('symbol', 'Unknown')
        if symbol not in contract_stats:
            contract_stats[symbol] = {
                'trades': 0,
                'wins': 0,
                'pnl': 0,
                'volume': 0
            }
        contract_stats[symbol]['trades'] += 1
        contract_stats[symbol]['wins'] += 1 if t.get('pnl', 0) > 0 else 0
        contract_stats[symbol]['pnl'] += t.get('pnl', 0)
        contract_stats[symbol]['volume'] += t.get('qty', 0)
    
    # Додати win_rate для кожного контракту
    for symbol in contract_stats:
        stats = contract_stats[symbol]
        stats['win_rate'] = round(stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0, 1)
        stats['avg_pnl'] = round(stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0, 2)
    
    # Сортувати по P&L (спадаючи)
    sorted_contracts = sorted(
        contract_stats.items(),
        key=lambda x: x[1]['pnl'],
        reverse=True
    )
    
    return {
        'total_trades': total,
        'win_trades': wins,
        'loss_trades': losses,
        'win_rate': win_rate,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': avg_pnl,
        'total_volume': round(total_volume, 2),
        'avg_trade_size': avg_trade_size,
        'best_trade': round(best, 2),
        'worst_trade': round(worst, 2),
        'consecutive_wins': wins_streak,
        'consecutive_losses': losses_streak,
        'contract_stats': sorted_contracts[:10]  # Топ 10 контрактів
    }


def get_trades_with_multiple_periods():
    """
    Отримує статистику по всіх періодам одночасно
    для уникнення множинних DB запитів
    """
    periods = [7, 30, 60, 90, 180]
    result = {}
    
    for days in periods:
        trades = stats_service.get_trades(days=days)
        result[days] = calculate_stats(trades)
    
    return result
