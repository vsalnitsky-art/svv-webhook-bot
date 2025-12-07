#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 AlgoBot - Профессиональная торговая платформа
Обновлено: 07.12.2025
"""

import sys
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from collections import defaultdict
import numpy as np

# ============================================================================
# WHALE HUNTER SCANNER
# ============================================================================
try:
    from whale_hunter_improved import init_scanner, scan_in_background, whale_hunter as wh
    WHALE_HUNTER_AVAILABLE = True
except Exception as e:
    WHALE_HUNTER_AVAILABLE = False
    print(f"⚠️  Whale Hunter Scanner не инициализирован: {e}")

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ WHALE HUNTER
# ============================================================================

if WHALE_HUNTER_AVAILABLE:
    try:
        wh_instance = init_scanner('1h')
        scan_in_background(interval=300)
        logger.info("✅ Whale Hunter Scanner инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации сканера: {e}")
        wh_instance = None
else:
    wh_instance = None

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def calculate_stats(trades):
    """
    Расчет подробной статистики торгов
    """
    if not trades:
        return {
            'total_trades': 0, 'win_trades': 0, 'loss_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            'total_volume': 0, 'avg_trade_size': 0,
            'best_trade': 0, 'worst_trade': 0,
            'consecutive_wins': 0, 'consecutive_losses': 0,
            'contract_stats': []
        }
    
    total = len(trades)
    wins = len([t for t in trades if t.get('pnl', 0) > 0])
    losses = total - wins
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = round((wins / total * 100) if total > 0 else 0, 1)
    avg_pnl = round(total_pnl / total if total > 0 else 0, 2)
    
    total_volume = sum(t.get('qty', 0) for t in trades)
    avg_trade_size = round(total_volume / total if total > 0 else 0, 2)
    
    best = max((t.get('pnl', 0) for t in trades), default=0)
    worst = min((t.get('pnl', 0) for t in trades), default=0)
    
    # Расчет стриков
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
    
    # Статистика по контрактам
    contract_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'volume': 0})
    
    for t in trades:
        symbol = t.get('symbol', 'Unknown')
        contract_stats[symbol]['trades'] += 1
        if t.get('pnl', 0) > 0:
            contract_stats[symbol]['wins'] += 1
        contract_stats[symbol]['pnl'] += t.get('pnl', 0)
        contract_stats[symbol]['volume'] += t.get('qty', 0)
    
    # Добавить win_rate для каждого контракта
    for symbol in contract_stats:
        trades_count = contract_stats[symbol]['trades']
        wins_count = contract_stats[symbol]['wins']
        contract_stats[symbol]['win_rate'] = round(
            (wins_count / trades_count * 100) if trades_count > 0 else 0, 1
        )
    
    # Сортировка по PnL
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
        'contract_stats': sorted_contracts
    }

# ============================================================================
# МАРШРУТЫ
# ============================================================================

@app.route('/', methods=['GET'])
def index_page():
    """Профессиональный обзор рынка"""
    try:
        days_param = int(request.args.get('days', 7))
    except:
        days_param = 7
    
    if days_param not in [7, 30, 60, 90, 180]:
        days_param = 7
    
    # Симуляция данных торгов
    trades = [
        {
            'symbol': 'BTCUSDT', 'side': 'Long', 'entry': 45000, 'exit': 45500,
            'qty': 1, 'pnl': 500, 'status': 'Closed', 'time': datetime.now().isoformat(),
            'commission': 0.001
        },
        {
            'symbol': 'ETHUSDT', 'side': 'Short', 'entry': 2500, 'exit': 2480,
            'qty': 10, 'pnl': 200, 'status': 'Closed', 'time': datetime.now().isoformat(),
            'commission': 0.001
        },
    ]
    
    # Получить баланс (симуляция)
    balance = 10000
    active_count = 2
    
    # Расчет статистики
    stats = calculate_stats(trades)
    period_pnl = stats['total_pnl']
    longs = sum(1 for t in trades if t.get('side') == 'Long')
    shorts = sum(1 for t in trades if t.get('side') == 'Short')
    
    return render_template('index.html',
                          date=datetime.utcnow().strftime('%d %b %Y'),
                          balance=balance,
                          active_count=active_count,
                          period_pnl=period_pnl,
                          longs=longs,
                          shorts=shorts,
                          days=days_param,
                          trades=trades[:15] if trades else [],
                          stats=stats)

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Монитор активных торгов"""
    return render_template('scanner.html')

@app.route('/analyzer', methods=['GET'])
def analyzer_page():
    """Анализатор рынка"""
    return render_template('analyzer.html')

# ============================================================================
# WHALE HUNTER SCANNER МАРШРУТЫ
# ============================================================================

@app.route('/api/whale_hunter/scan', methods=['POST'])
def whale_hunter_scan():
    """Запустить сканирование"""
    try:
        if not WHALE_HUNTER_AVAILABLE or wh_instance is None:
            return jsonify({'status': 'error', 'message': 'Scanner not available'}), 500
        
        signals = wh_instance.scan_market()
        return jsonify({
            'status': 'success',
            'signals_found': len(signals),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Scan error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/signals', methods=['GET'])
def whale_hunter_signals():
    """Получить активные сигналы"""
    try:
        if not WHALE_HUNTER_AVAILABLE or wh_instance is None:
            return jsonify({'signals': []})
        
        signals = wh_instance.get_signals()
        return jsonify({
            'status': 'success',
            'count': len(signals),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/history', methods=['GET'])
def whale_hunter_history():
    """История сигналов"""
    try:
        if not WHALE_HUNTER_AVAILABLE or wh_instance is None:
            return jsonify({'history': []})
        
        limit = request.args.get('limit', 100, type=int)
        history = wh_instance.get_history(limit=limit)
        return jsonify({
            'status': 'success',
            'count': len(history),
            'history': history,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/status', methods=['GET'])
def whale_hunter_status():
    """Статус сканера"""
    try:
        if not WHALE_HUNTER_AVAILABLE or wh_instance is None:
            return jsonify({
                'status': 'not_available',
                'active_signals': 0,
                'last_scan': None
            })
        
        return jsonify({
            'status': 'running',
            'timeframe': wh_instance.TIMEFRAME,
            'active_signals': len(wh_instance.get_signals()),
            'total_history': len(wh_instance.get_history()),
            'min_volume': wh_instance.MIN_VOLUME,
            'last_scan': wh_instance.signal_history[-1].get('timestamp') if wh_instance.signal_history else None
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/settings', methods=['GET', 'POST'])
def whale_hunter_settings():
    """Получить/изменить настройки сканера"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            timeframe = data.get('timeframe', '1h')
            min_volume = data.get('min_volume', 5000000)
            
            global wh_instance
            if WHALE_HUNTER_AVAILABLE:
                wh_instance = init_scanner(timeframe)
                wh_instance.MIN_VOLUME = min_volume
            
            return jsonify({'status': 'success', 'message': 'Settings updated'})
        
        if not WHALE_HUNTER_AVAILABLE or wh_instance is None:
            return jsonify({
                'timeframe': '1h',
                'min_volume': 5000000
            })
        
        return jsonify({
            'timeframe': wh_instance.TIMEFRAME,
            'min_volume': wh_instance.MIN_VOLUME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# СТРАНИЦА СТРАТЕГИИ (ОЧЕНЬ ВАЖНО!)
# ============================================================================

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def strategy_page():
    """Страница Whale Hunter Scanner"""
    try:
        if request.method == 'POST':
            data = request.get_json() or request.form
            timeframe = data.get('timeframe', '1h')
            
            global wh_instance
            if WHALE_HUNTER_AVAILABLE:
                wh_instance = init_scanner(timeframe)
            
            logger.info(f"⚙️ Стратегия обновлена: TF={timeframe}")
            
            if request.is_json:
                return jsonify({'status': 'success'})
            return redirect(url_for('strategy_page'))
        
        # GET - показать страницу
        status = {
            'timeframe': wh_instance.TIMEFRAME if (WHALE_HUNTER_AVAILABLE and wh_instance) else '1h',
            'active_signals': len(wh_instance.get_signals()) if (WHALE_HUNTER_AVAILABLE and wh_instance) else 0,
            'last_scan': wh_instance.signal_history[-1].get('timestamp') if (WHALE_HUNTER_AVAILABLE and wh_instance and wh_instance.signal_history) else None
        }
        
        return render_template('strategy_ob_trend.html', status=status)
    except Exception as e:
        logger.error(f"❌ Strategy page error: {e}")
        return render_template('strategy_ob_trend.html', error=str(e))

# ============================================================================
# ОБРАБОТКА ОШИБОК
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '404 Not Found', 'message': 'The requested URL was not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({'error': '500 Internal Server Error', 'message': str(error)}), 500

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == '__main__':
    logger.info("🚀 AlgoBot запускается...")
    logger.info(f"✅ Whale Hunter Scanner: {'Доступен' if WHALE_HUNTER_AVAILABLE else 'Недоступен'}")
    app.run(host='0.0.0.0', port=10000, debug=False)
