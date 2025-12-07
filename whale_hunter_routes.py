"""
FLASK МАРШРУТИ ДЛЯ WHALE HUNTER SCANNER
Додайте це в main_app.py
"""

from whale_hunter_improved import init_scanner, scan_in_background, whale_hunter
import json
from flask import jsonify, request

# ============================================================================
# ІНІЦІАЛІЗАЦІЯ СКАНЕРА
# ============================================================================

def init_whale_hunter():
    """Ініціалізація при запуску Flask"""
    try:
        global whale_hunter
        scanner = init_scanner('1h')
        # Запустити фоновое сканування
        scan_in_background(interval=300)  # Кожні 5 хвилин
        logger.info("✅ Whale Hunter Scanner ініціалізовано")
    except Exception as e:
        logger.error(f"❌ Помилка ініціалізації сканера: {e}")

# Запустити при загрузці
init_whale_hunter()

# ============================================================================
# МАРШРУТИ
# ============================================================================

@app.route('/api/whale_hunter/settings', methods=['GET', 'POST'])
def whale_hunter_settings():
    """Налаштування сканера"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            timeframe = data.get('timeframe', '1h')
            min_volume = data.get('min_volume', 5000000)
            
            # Оновити глобальний сканер
            global whale_hunter
            whale_hunter = init_scanner(timeframe)
            whale_hunter.MIN_VOLUME = min_volume
            
            logger.info(f"⚙️ Оновлено: TF={timeframe}, MinVol={min_volume}")
            
            return jsonify({
                'status': 'success',
                'message': 'Налаштування оновлено',
                'timeframe': timeframe,
                'min_volume': min_volume
            })
        
        # GET - повернути поточні налаштування
        return jsonify({
            'timeframe': whale_hunter.TIMEFRAME if whale_hunter else '1h',
            'min_volume': whale_hunter.MIN_VOLUME if whale_hunter else 5000000,
            'bb_length': 20,
            'bb_std': 2.0,
            'adx_threshold': 25,
            'ichimoku_tenkan': 9,
            'ichimoku_kijun': 26
        })
    except Exception as e:
        logger.error(f"❌ Settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/scan', methods=['POST'])
def whale_hunter_scan_now():
    """Запустити сканування ЗАРАЗ"""
    try:
        if whale_hunter is None:
            return jsonify({'status': 'error', 'message': 'Scanner not initialized'}), 500
        
        signals = whale_hunter.scan_market()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'signals_found': len(signals),
            'signals': signals
        })
    except Exception as e:
        logger.error(f"❌ Scan error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/signals', methods=['GET'])
def whale_hunter_get_signals():
    """Отримати поточні активні сигнали"""
    try:
        if whale_hunter is None:
            return jsonify({'signals': []})
        
        signals = whale_hunter.get_signals()
        
        return jsonify({
            'status': 'success',
            'count': len(signals),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Get signals error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/history', methods=['GET'])
def whale_hunter_get_history():
    """Отримати історію сигналів"""
    try:
        if whale_hunter is None:
            return jsonify({'history': []})
        
        limit = request.args.get('limit', 100, type=int)
        history = whale_hunter.get_history(limit=limit)
        
        return jsonify({
            'status': 'success',
            'count': len(history),
            'history': history,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Get history error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/whale_hunter/status', methods=['GET'])
def whale_hunter_status():
    """Статус сканера"""
    try:
        if whale_hunter is None:
            return jsonify({
                'status': 'not_initialized',
                'active_signals': 0,
                'last_scan': None
            })
        
        return jsonify({
            'status': 'running',
            'timeframe': whale_hunter.TIMEFRAME,
            'active_signals': len(whale_hunter.get_signals()),
            'total_history': len(whale_hunter.get_history()),
            'min_volume': whale_hunter.MIN_VOLUME,
            'last_scan': whale_hunter.signal_history[-1].get('timestamp') if whale_hunter.signal_history else None
        })
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def strategy_whale_hunter():
    """Сторінка стратегії (заменить старый маршрут)"""
    try:
        if request.method == 'POST':
            # Обробка форми налаштувань
            timeframe = request.form.get('timeframe', '1h')
            scan_interval = int(request.form.get('scan_interval', 300))
            
            global whale_hunter
            whale_hunter = init_scanner(timeframe)
            
            logger.info(f"⚙️ Стратегія оновлена: TF={timeframe}")
            return redirect(url_for('strategy_whale_hunter'))
        
        # GET - показати сторінку
        status = {
            'timeframe': whale_hunter.TIMEFRAME if whale_hunter else '1h',
            'active_signals': len(whale_hunter.get_signals()) if whale_hunter else 0,
            'last_scan': whale_hunter.signal_history[-1].get('timestamp') if whale_hunter and whale_hunter.signal_history else None,
            'signals': whale_hunter.get_signals() if whale_hunter else []
        }
        
        return render_template('strategy_ob_trend.html', status=status)
    except Exception as e:
        logger.error(f"❌ Strategy page error: {e}")
        return render_template('strategy_ob_trend.html', error=str(e))

# ============================================================================
# WEBHOOK ДЛЯ ТОРГІВЕЛЬНОГО БОТУTO INTEGRATE WITH SIGNALS
# ============================================================================

@app.route('/api/whale_hunter/webhook', methods=['POST'])
def whale_hunter_webhook():
    """Webhook для передачі сигналів у торгівельний бот"""
    try:
        data = request.get_json()
        signal = data
        
        # Тут можна додати логіку для автоматичного входу
        logger.info(f"🐋 Webhook сигнал: {signal['symbol']} | Entry: {signal['entry']}")
        
        return jsonify({
            'status': 'received',
            'signal': signal['symbol'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
