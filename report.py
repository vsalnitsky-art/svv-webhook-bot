"""
Report Module - P&L UI 📊
Відповідає за рендеринг сторінки статистики.
"""
from flask import render_template_string
from statistics_service import stats_service
from datetime import datetime, timedelta

def render_report_page(bot_instance, request):
    # 1. Отримуємо параметри
    days = int(request.args.get('days', 7))
    s_arg, e_arg = request.args.get('start'), request.args.get('end')
    
    # 2. Синхронізуємо дані з біржі (через бот)
    bot_instance.sync_trades(days)
    
    # 3. Отримуємо баланс
    bal = bot_instance.get_available_balance() or 0.0
    
    # 4. Отримуємо статистику з бази
    try:
        trades = stats_service.get_trades(90)
        if not trades: raise ValueError("No trades")
        
        filtered = []
        s_dt, e_dt = None, None
        if s_arg and e_arg:
            s_dt = datetime.strptime(s_arg, '%Y-%m-%d')
            e_dt = datetime.strptime(e_arg, '%Y-%m-%d') + timedelta(days=1)
        elif days:
            e_dt = datetime.now()
            s_dt = e_dt - timedelta(days=days)
            
        for t in trades:
            if not t['exit_time']: continue
            et = datetime.strptime(t['exit_time'], '%d.%m %H:%M') if isinstance(t['exit_time'], str) else t['exit_time']
            et = et.replace(year=datetime.now().year)
            if s_dt and e_dt:
                if s_dt <= et <= e_dt: filtered.append(t)
            else: filtered.append(t)
            
        stats = {"total_trades": len(filtered), "total_pnl": 0.0, "total_volume": 0.0, "win_trades": 0, "loss_trades": 0, "long_trades": 0, "short_trades": 0, "long_pnl":0, "short_pnl":0, "details": [], "chart_labels": [], "chart_data": [], "coin_performance":{}}
        
        filtered.sort(key=lambda x: x['exit_time'], reverse=False)
        run_bal = 0
        daily = {}
        
        for t in filtered:
            stats["total_pnl"] += t['pnl']
            run_bal += t['pnl']
            stats["total_volume"] += t.get('qty',0)*t.get('exit_price',0)
            if t['pnl']>0: stats["win_trades"]+=1
            else: stats["loss_trades"]+=1
            
            if t['side'] == 'Long': 
                stats['long_trades'] += 1; stats['long_pnl'] += t['pnl']
            else: 
                stats['short_trades'] += 1; stats['short_pnl'] += t['pnl']
            
            sym = t['symbol']
            if sym not in stats['coin_performance']: stats['coin_performance'][sym] = 0.0
            stats['coin_performance'][sym] += t['pnl']
            
            d_str = t['exit_time'].split(' ')[0]
            daily[d_str] = daily.get(d_str, 0) + t['pnl']
            stats["details"].append(t)
        
        rb = 0
        for d in sorted(daily.keys()):
            rb += daily[d]
            stats["chart_labels"].append(d)
            stats["chart_data"].append(round(rb, 2))
            
        top = sorted(stats['coin_performance'].items(), key=lambda x: x[1], reverse=True)
        stats['top_coins_labels'] = [x[0] for x in top[:5]]
        stats['top_coins_values'] = [round(x[1], 2) for x in top[:5]]
        stats["details"].sort(key=lambda x: x['exit_time'], reverse=True)
        if stats["total_trades"]>0: stats["win_rate"] = round((stats["win_trades"]/stats["total_trades"])*100,1)
        
    except:
        stats = {"total_pnl":0, "win_rate":0, "total_trades":0, "volume":0, "chart_labels":[], "chart_data":[], "details":[], "long_trades":0, "short_trades":0, "win_trades":0, "loss_trades":0, "top_coins_labels":[], "top_coins_values":[], "long_pnl":0, "short_pnl":0}

    # HTML ТЕМПЛЕЙТ (BYBIT STYLE)
    html = """
    <!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><title>P&L Analysis</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet"><style>
    :root { --bg-color: #ffffff; --text-primary: #121214; --text-secondary: #858e9c; --green: #20b26c; --red: #ef454a; --btn-active-bg: #fff8d9; --btn-active-text: #cf9e04; --border: #f4f4f4; }
    body { font-family: 'Roboto', sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; padding: 20px; }
    .container { max-width: 1280px; margin: 0 auto; }
    .header { display: flex; align-items: center; margin-bottom: 30px; }
    .title { font-size: 20px; font-weight: 700; margin-right: 20px; }
    .btn-group { display: flex; gap: 10px; }
    .btn { border: none; background: none; padding: 6px 12px; border-radius: 4px; font-size: 13px; cursor: pointer; color: var(--text-primary); font-weight: 500; text-decoration:none; }
    .btn:hover { background: #f5f5f5; }
    .btn.active { background-color: var(--btn-active-bg); color: var(--btn-active-text); }
    .summary-grid { display: flex; gap: 60px; margin-bottom: 30px; }
    .stat-item { display: flex; flex-direction: column; }
    .stat-label { font-size: 12px; color: var(--text-secondary); margin-bottom: 5px; text-decoration: underline dotted; cursor: help; }
    .stat-value { font-size: 28px; font-weight: 700; }
    .text-green { color: var(--green); } .text-red { color: var(--red); }
    .charts-container { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin-bottom: 40px; }
    .bottom-stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 20px; margin-bottom: 40px; }
    .b-stat-box { padding: 15px 0; }
    .b-stat-header { font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; }
    .b-stat-val { font-size: 24px; font-weight: 700; }
    .custom-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .custom-table th { text-align: left; color: var(--text-secondary); font-weight: 400; padding: 10px 0; border-bottom: 1px solid var(--border); }
    .custom-table td { padding: 14px 0; border-bottom: 1px solid var(--border); vertical-align: middle; }
    .badge { padding: 2px 6px; border-radius: 2px; font-size: 11px; }
    .badge-success { background: #fff8ec; color: #cf9e04; } .badge-loss { background: #f5f5f5; color: #858e9c; }
    .type-long { color: var(--green); } .type-short { color: var(--red); }
    </style></head><body>
    <div class="container"><div class="header"><div class="title">P&L</div><div class="btn-group"><a href="/report?days=7" class="btn {{ 'active' if days==7 }}">7 дн.</a><a href="/report?days=30" class="btn {{ 'active' if days==30 }}">30 дн.</a><a href="/scanner" class="btn">← Сканер</a></div></div>
    <div class="summary-grid"><div class="stat-item"><div class="stat-label">Общий P&L</div><div class="stat-value {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">{{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} USD</div></div><div class="stat-item"><div class="stat-label">Объем</div><div class="stat-value text-green">{{ "{:,.0f}".format(stats.total_volume) }} USD</div></div></div>
    <div class="charts-container"><div class="chart-box"><div style="height: 300px;"><canvas id="pnlChart"></canvas></div></div><div class="chart-box"><div style="height: 300px;"><canvas id="rankChart"></canvas></div></div></div>
    <div class="bottom-stats"><div class="b-stat-box"><div class="b-stat-header">Всего ордеров</div><div class="b-stat-val">{{ stats.total_trades }}</div></div><div class="b-stat-box"><div class="b-stat-header">Успешных</div><div class="b-stat-val">{{ stats.win_rate }} %</div></div><div class="b-stat-box"><div class="b-stat-header">P&L Long</div><div class="b-stat-val {{ 'text-green' if stats.long_pnl >= 0 else 'text-red' }}">{{ "%.2f"|format(stats.long_pnl) }}</div></div><div class="b-stat-box"><div class="b-stat-header">P&L Short</div><div class="b-stat-val {{ 'text-green' if stats.short_pnl >= 0 else 'text-red' }}">{{ "%.2f"|format(stats.short_pnl) }}</div></div></div>
    <table class="custom-table"><thead><tr><th>Контракт</th><th>Тип</th><th>P&L</th><th>Результат</th><th>Время</th></tr></thead><tbody>
    {% for t in stats.details %}<tr><td style="font-weight: 500;">{{ t.symbol }}</td><td class="{{ 'type-long' if t.side == 'Long' else 'type-short' }}">{{ "Лонг" if t.side == 'Long' else "Шорт" }}</td><td class="{{ 'text-red' if t.pnl < 0 else 'text-green' }}">{{ "+" if t.pnl > 0 }}{{ "%.4f"|format(t.pnl) }}</td><td><span class="badge {{ 'badge-success' if t.pnl > 0 else 'badge-loss' }}">{{ "Успех" if t.pnl > 0 else "Убыток" }}</span></td><td style="color: var(--text-secondary);">{{ t.exit_time }}</td></tr>{% endfor %}
    </tbody></table></div>
    <script>
    const ctx = document.getElementById('pnlChart').getContext('2d'); const gradient = ctx.createLinearGradient(0, 0, 0, 300); gradient.addColorStop(0, 'rgba(239, 69, 74, 0.2)'); gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    new Chart(ctx, {type: 'line', data: {labels: {{ stats.chart_labels|tojson }}, datasets: [{data: {{ stats.chart_data|tojson }}, borderColor: '#ef454a', backgroundColor: gradient, borderWidth: 2, pointRadius: 0, fill: true}]}, options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {grid: {display: false}}, y: {grid: {color: '#f4f4f4'}}}}});
    const ctxBar = document.getElementById('rankChart').getContext('2d'); new Chart(ctxBar, {type: 'bar', data: {labels: {{ stats.top_coins_labels|tojson }}, datasets: [{data: {{ stats.top_coins_values|tojson }}, backgroundColor: (ctx) => ctx.raw >= 0 ? '#20b26c' : '#ef454a', borderRadius: 2}]}, options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {display: false}, y: {grid: {display: false}}}}});
    </script></body></html>
    """
    return render_template_string(html, stats=stats, bal=bal, days=days)
