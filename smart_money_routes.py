#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SMART MONEY API ROUTES v2.0
==============================
API endpoints для Smart Money Engine
"""

from flask import Blueprint, render_template, request, jsonify
from smart_money_engine import smart_money_engine
import logging

logger = logging.getLogger("SmartMoneyRoutes")

# Blueprint для модульності
smart_money_bp = Blueprint('smart_money_v2', __name__, url_prefix='/smart_money')


# ============================================================================
#                              PAGE ROUTES
# ============================================================================

@smart_money_bp.route('')
def smart_money_page():
    """Головна сторінка Smart Money v2"""
    return render_template('smart_money.html')


# ============================================================================
#                              API: WATCHLIST
# ============================================================================

@smart_money_bp.route('/api/watchlist', methods=['GET'])
def api_get_watchlist():
    """Отримати watchlist"""
    return jsonify(smart_money_engine.get_watchlist())


@smart_money_bp.route('/api/watchlist/add', methods=['POST'])
def api_add_to_watchlist():
    """Додати до watchlist"""
    data = request.get_json() or {}
    symbol = data.get('symbol', '').upper().strip()
    direction = data.get('direction', 'BUY')
    source = data.get('source', 'Manual')
    
    if not symbol:
        return jsonify({'status': 'error', 'error': 'Symbol required'}), 400
    
    result = smart_money_engine.add_to_watchlist(symbol, direction, source)
    return jsonify(result)


@smart_money_bp.route('/api/watchlist/remove/<symbol>', methods=['POST'])
def api_remove_from_watchlist(symbol):
    """Видалити з watchlist"""
    result = smart_money_engine.remove_from_watchlist(symbol)
    return jsonify(result)


@smart_money_bp.route('/api/watchlist/clear', methods=['POST'])
def api_clear_watchlist():
    """Очистити watchlist"""
    result = smart_money_engine.clear_watchlist()
    return jsonify(result)


# ============================================================================
#                           API: DETECTED OBs
# ============================================================================

@smart_money_bp.route('/api/detected', methods=['GET'])
def api_get_detected():
    """Отримати detected OBs"""
    return jsonify(smart_money_engine.get_detected_obs())


@smart_money_bp.route('/api/detected/delete/<int:ob_id>', methods=['POST'])
def api_delete_detected(ob_id):
    """Видалити detected OB"""
    result = smart_money_engine.delete_detected_ob(ob_id)
    return jsonify(result)


@smart_money_bp.route('/api/detected/clear', methods=['POST'])
def api_clear_detected():
    """Очистити всі detected OBs"""
    result = smart_money_engine.clear_detected_obs()
    return jsonify(result)


# ============================================================================
#                          API: EXECUTION LOG
# ============================================================================

@smart_money_bp.route('/api/execution_log', methods=['GET'])
def api_get_execution_log():
    """Отримати execution log"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify(smart_money_engine.get_execution_log(limit))


@smart_money_bp.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Отримати статистику"""
    return jsonify(smart_money_engine.get_execution_stats())


@smart_money_bp.route('/api/execution_log/<int:log_id>', methods=['DELETE'])
def api_delete_execution_log(log_id):
    """Видалити запис з Execution Log"""
    return jsonify(smart_money_engine.delete_execution_log(log_id))


@smart_money_bp.route('/api/execution_log/clear', methods=['POST'])
def api_clear_execution_log():
    """Очистити весь Execution Log"""
    return jsonify(smart_money_engine.clear_execution_log())


# ============================================================================
#                            API: CONFIG
# ============================================================================

@smart_money_bp.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Отримати/зберегти конфігурацію"""
    if request.method == 'POST':
        data = request.get_json() or {}
        smart_money_engine.save_config(data)
        
        # Перезапуск auto scan якщо потрібно
        config = smart_money_engine.get_config()
        if config.get('ob_auto_scan'):
            smart_money_engine.start_auto_scan()
        else:
            smart_money_engine.stop_auto_scan()
        
        # Перезапуск exit monitor якщо потрібно
        if config.get('ob_exit_enabled'):
            smart_money_engine.start_exit_monitor()
        else:
            smart_money_engine.stop_exit_monitor()
        
        return jsonify({'status': 'ok'})
    
    return jsonify(smart_money_engine.get_config())


@smart_money_bp.route('/api/param_help', methods=['GET'])
def api_param_help():
    """Отримати довідку по параметрах"""
    return jsonify(smart_money_engine.get_param_help())


# ============================================================================
#                            API: SCANNING
# ============================================================================

@smart_money_bp.route('/api/scan', methods=['POST'])
def api_scan():
    """Запустити сканування"""
    import threading
    
    def run_scan():
        smart_money_engine.scan_watchlist()
    
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({'status': 'started'})


@smart_money_bp.route('/api/scan/stop', methods=['POST'])
def api_stop_scan():
    """Зупинити сканування"""
    smart_money_engine._stop_scan.set()
    return jsonify({'status': 'stopped'})


@smart_money_bp.route('/api/status', methods=['GET'])
def api_status():
    """Отримати статус системи"""
    return jsonify(smart_money_engine.get_status())


# ============================================================================
#                          API: AUTO SCAN
# ============================================================================

@smart_money_bp.route('/api/auto_scan/start', methods=['POST'])
def api_start_auto_scan():
    """Запустити auto scan"""
    smart_money_engine.start_auto_scan()
    return jsonify({'status': 'started'})


@smart_money_bp.route('/api/auto_scan/stop', methods=['POST'])
def api_stop_auto_scan():
    """Зупинити auto scan"""
    smart_money_engine.stop_auto_scan()
    return jsonify({'status': 'stopped'})


# ============================================================================
#                         API: EXIT MONITOR
# ============================================================================

@smart_money_bp.route('/api/exit_monitor/start', methods=['POST'])
def api_start_exit_monitor():
    """Запустити exit monitor"""
    smart_money_engine.start_exit_monitor()
    return jsonify({'status': 'started'})


@smart_money_bp.route('/api/exit_monitor/stop', methods=['POST'])
def api_stop_exit_monitor():
    """Зупинити exit monitor"""
    smart_money_engine.stop_exit_monitor()
    return jsonify({'status': 'stopped'})


# ============================================================================
#                       API: COORDINATOR STATUS
# ============================================================================

@smart_money_bp.route('/api/coordinator/status', methods=['GET'])
def api_coordinator_status():
    """Отримати статус координатора сканерів"""
    try:
        from scanner_coordinator import scanner_coordinator
        return jsonify(scanner_coordinator.get_status())
    except ImportError:
        return jsonify({'error': 'Coordinator not available'})


@smart_money_bp.route('/api/coordinator/trigger/<scanner_type>', methods=['POST'])
def api_coordinator_trigger(scanner_type):
    """Примусово запустити сканер через координатор"""
    try:
        from scanner_coordinator import scanner_coordinator, ScannerType
        
        scanner_map = {
            'smart_money': ScannerType.SMART_MONEY,
            'whale_hunter': ScannerType.WHALE_HUNTER,
            'whale_pro': ScannerType.WHALE_PRO,
            'whale_sniper': ScannerType.WHALE_SNIPER,
            'rsi_mfi': ScannerType.RSI_MFI
        }
        
        st = scanner_map.get(scanner_type)
        if not st:
            return jsonify({'status': 'error', 'error': 'Unknown scanner type'})
        
        result = scanner_coordinator.trigger_scan(st)
        return jsonify({'status': 'ok' if result else 'busy'})
        
    except ImportError:
        return jsonify({'error': 'Coordinator not available'})


# ============================================================================
#                          REGISTER ROUTES
# ============================================================================

def register_smart_money_routes(app):
    """Реєструє routes в Flask app"""
    app.register_blueprint(smart_money_bp)
    logger.info("✅ Smart Money v2 routes registered")
    
    # Автозапуск якщо увімкнено
    config = smart_money_engine.get_config()
    if config.get('ob_auto_scan'):
        smart_money_engine.start_auto_scan()
    if config.get('ob_exit_enabled'):
        smart_money_engine.start_exit_monitor()
