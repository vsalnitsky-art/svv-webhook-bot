"""
Report Module - Professional Bybit-style P&L Analytics
"""
from flask import render_template, request
from models import db_manager, Trade
from sqlalchemy import desc
from datetime import datetime, timedelta
import json

def render_report_page(bot_instance, request):
    days_filter = int(request.args.get('days', 7))
    
    # Спроба синхронізації
    try:
        bot_instance.sync_trades(days=days_filter)
    except Exception as e:
        print(f"Sync warning: {e}")

    session = db_manager.get_session()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_filter)
        trades = session.query(Trade).filter(Trade.exit_time >= cutoff_date).order_by(Trade.exit_time.asc()).all()
        
        total_pnl = 0.0
        total_volume = 0.0
        wins = 0
        losses = 0
        cnt_long = 0
        cnt_short = 0
        
        chart_labels = []       
        chart_equity = []       
        chart_daily_pnl = {}    
        
        cumulative_pnl = 0
        
        for t in trades:
            pnl = t.pnl
            total_pnl += pnl
            cumulative_pnl += pnl
            
            vol = (t.entry_price * t.qty) if t.entry_price and t.qty else 0
            total_volume += vol
            
            if pnl > 0: wins += 1
            else: losses += 1
            
            if t.side == 'Long': cnt_long += 1
            else: cnt_short += 1
            
            date_str = t.exit_time.strftime('%Y-%m-%d %H:%M')
            chart_labels.append(date_str)
            chart_equity.append(round(cumulative_pnl, 2))
            
            day_key = t.exit_time.strftime('%Y-%m-%d')
            chart_daily_pnl[day_key] = chart_daily_pnl.get(day_key, 0) + pnl

        total_trades = len(trades)
        win_rate = round((wins / total_trades * 100), 1) if total_trades > 0 else 0
        avg_profit = round(total_pnl / total_trades, 2) if total_trades > 0 else 0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99 if gross_profit > 0 else 0)

        daily_labels = sorted(chart_daily_pnl.keys())
        daily_values = [round(chart_daily_pnl[k], 2) for k in daily_labels]

        stats = {
            'total_pnl': round(total_pnl, 2),
            'win_rate': win_rate,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'volume': round(total_volume, 2),
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'longs': cnt_long,
            'shorts': cnt_short
        }
        
        recent_trades = sorted(trades, key=lambda x: x.exit_time, reverse=True)

        return render_template('report.html', 
                               stats=stats, 
                               trades=recent_trades, 
                               days=days_filter,
                               chart_labels=json.dumps(chart_labels),
                               chart_equity=json.dumps(chart_equity),
                               daily_labels=json.dumps(daily_labels),
                               daily_values=json.dumps(daily_values))
        
    except Exception as e:
        print(f"Report Error: {e}")
        return f"Error generating report: {e}"
    finally:
        session.close()