"""
Main App - Clean & Professional
Updated with proper logging and self-ping mechanism
"""
import logging
import threading
import time
import json
import ctypes
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import requests

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from scanner_config import ScannerConfig
from report import render_report_page
from models import db_manager, MarketCandidate  # ⭐ НОВОЕ для навигации

# Запобігання сну у Windows (якщо запускається локально)
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Сканер для моніторингу
scanner_config = ScannerConfig()
scanner = EnhancedMarketScanner(bot_instance, scanner_config)
scanner.start()

# Монітор для запису логів в базу
def monitor_active():
    """Фоновий потік для запису стану позицій в БД"""
    logger.info("Starting active position monitor...")
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                for p in r['result']['list']:
                    if float(p['size']) > 0:
                        stats_service.save_monitor_log({
                            'symbol': p['symbol'], 
                            'price': float(p['avgPrice']), 
                            'pnl': float(p['unrealisedPnl']), 
                            'rsi': scanner.get_current_rsi(p['symbol']), 
                            'pressure': scanner.get_market_pressure(p['symbol'])
                        })
            else:
                logger.warning(f"Monitor Warning: {r.get('retMsg')}")
        except Exception as e:
            logger.error(f"Error in monitor_active loop: {e}")
        
        time.sleep(10)

threading.Thread(target=monitor_active, daemon=True).start()

def keep_alive():
    """
    Механізм запобігання засипанню (Self-Ping).
    Пінгує сам себе кожні N хвилин (управляется через scanner_config).
    """
    time.sleep(5)
    
    # Проверяем включен ли keep-alive
    if not scanner_config.keep_alive_enabled:
        logger.info("💤 Keep-alive ВИМКНЕНО в конфігурації")
        return
    
    external_url = os.environ.get('RENDER_EXTERNAL_URL')
    local_url = f'http://127.0.0.1:{config.PORT}/health'
    
    target_url = f"{external_url}/health" if external_url else local_url
    interval = scanner_config.keep_alive_interval
    
    logger.info(f"💓 Сервіс Keep-alive запущено. Target: {target_url}, Interval: {interval}s")

    while scanner_config.keep_alive_enabled:  # ⭐ Проверяем на каждой итерации
        try:
            response = requests.get(target_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"💓 Self-Ping OK: {target_url}")
            else:
                logger.warning(f"⚠️ Self-Ping returned status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Self-Ping Failed: {e}")
        
        time.sleep(interval)  # ⭐ Используем interval из конфига
    
    logger.info("💤 Keep-alive зупинено (disabled in config)")

threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        # Логування дії (Close або Buy/Sell)
        logger.info(f"🔔 SIGNAL RECEIVED: {data.get('symbol')} {data.get('action')}")
        
        result = bot_instance.place_order(data)
        
        # Обробка статусів, що повернув bot.py
        if result.get("status") in ["ok", "ignored"]:
            return jsonify(result)
        else:
            logger.error(f"Order action failed: {result}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Scanner v2.0 - Enhanced UI with full position details"""
    positions = []
    
    try:
        # Получить активные позиции с биржи
        response = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        
        if response['retCode'] != 0:
            logger.error(f"Scanner Page API Error: {response.get('retMsg')}")
            return render_template('scanner.html', 
                                 positions=[], 
                                 active_count=0,
                                 total_auto_closes=0,
                                 success_rate=0,
                                 last_update='Error',
                                 update_time=datetime.now().strftime('%H:%M:%S'))
        
        # Обработать каждую позицию
        for p in response['result']['list']:
            if float(p['size']) <= 0:
                continue
            
            symbol = p['symbol']
            side = p['side']  # Buy=Long, Sell=Short
            
            # Получить информацию от PositionMonitor
            position_info = scanner.position_monitor.get_position_info(symbol)
            
            # Если позиция новая, информации может не быть
            if not position_info:
                position_info = {
                    'open_time': datetime.now(),
                    'hold_duration': 0,
                    'max_pnl': float(p['unrealisedPnl']),
                    'min_pnl': float(p['unrealisedPnl']),
                    'avg_rsi': 50.0,
                    'rsi_range': 'N/A',
                    'avg_mfi': 50.0,
                    'signal_count': 0,
                    'last_signal': 'Нет',
                }
            
            # Получить данные из активных позиций сканера
            active_data = scanner.position_monitor.active_positions.get(symbol, {})
            
            # Рассчитать текущие значения индикаторов
            rsi_value = active_data.get('rsi_values', [50])[-1] if active_data.get('rsi_values') else 50.0
            mfi_value = active_data.get('mfi_values', [50])[-1] if active_data.get('mfi_values') else 50.0
            
            # Определить цвет RSI
            if rsi_value < 30:
                rsi_color = 'rsi-green'
                rsi_text_color = 'var(--green)'
            elif rsi_value > 70:
                rsi_color = 'rsi-red'
                rsi_text_color = 'var(--red)'
            else:
                rsi_color = 'rsi-yellow'
                rsi_text_color = 'var(--yellow)'
            
            # Определить тренд MFI
            # Упрощённая логика - в реальной версии нужно получать из индикатора
            mfi_trend = 'Бычий' if mfi_value > 50 else ('Медвежий' if mfi_value < 50 else 'Нейтральный')
            mfi_trend_class = 'bullish' if mfi_value > 50 else ('bearish' if mfi_value < 50 else 'neutral')
            
            # Определить сигнал и его стиль
            current_signal = active_data.get('last_signal', 'Нет')
            if current_signal == 'Нет' or not current_signal:
                signal_class = ''
                signal_icon = ''
                signal_title = ''
            elif 'bullish' in current_signal.lower() or 'buy' in current_signal.lower():
                signal_class = 'success'
                signal_icon = '🟢'
                signal_title = 'Bullish Signal'
            elif 'bearish' in current_signal.lower() or 'sell' in current_signal.lower():
                signal_class = 'danger'
                signal_icon = '🔴'
                signal_title = 'Bearish Signal'
            else:
                signal_class = 'warning'
                signal_icon = '🟡'
                signal_title = 'Signal'
            
            # Рассчитать P&L процент
            entry_price = float(p['avgPrice'])
            current_price = float(p['markPrice'])
            size = float(p['size'])
            pnl = float(p['unrealisedPnl'])
            
            if entry_price > 0 and size > 0:
                pnl_percent = (pnl / (entry_price * size)) * 100
                max_pnl_percent = (position_info['max_pnl'] / (entry_price * size)) * 100
                min_pnl_percent = (position_info['min_pnl'] / (entry_price * size)) * 100
            else:
                pnl_percent = 0
                max_pnl_percent = 0
                min_pnl_percent = 0
            
            # Форматировать время удержания
            hold_seconds = position_info['hold_duration']
            if hold_seconds < 60:
                hold_duration = f"{int(hold_seconds)}s"
            elif hold_seconds < 3600:
                hold_duration = f"{int(hold_seconds / 60)}m {int(hold_seconds % 60)}s"
            else:
                hours = int(hold_seconds / 3600)
                minutes = int((hold_seconds % 3600) / 60)
                hold_duration = f"{hours}h {minutes}m"
            
            # Momentum
            momentum = '📈 Растёт' if rsi_value > 55 else ('📉 Падает' if rsi_value < 45 else '➡️ Флэт')
            
            # Добавить позицию в список
            positions.append({
                'symbol': symbol,
                'side': side,
                'size': p['size'],
                'leverage': p['leverage'],
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'max_pnl': position_info['max_pnl'],
                'max_pnl_percent': max_pnl_percent,
                'min_pnl': position_info['min_pnl'],
                'min_pnl_percent': min_pnl_percent,
                'open_time': position_info['open_time'].strftime('%Y-%m-%d %H:%M'),
                'hold_duration': hold_duration,
                'rsi_value': rsi_value,
                'rsi_color': rsi_color,
                'rsi_text_color': rsi_text_color,
                'mfi_value': mfi_value,
                'mfi_trend': mfi_trend,
                'mfi_trend_class': mfi_trend_class,
                'avg_rsi': position_info['avg_rsi'],
                'rsi_range': position_info['rsi_range'],
                'avg_mfi': position_info['avg_mfi'],
                'signal_count': position_info['signal_count'],
                'current_signal': current_signal,
                'signal_class': signal_class,
                'signal_icon': signal_icon,
                'signal_title': signal_title,
                'volume_24h': 'N/A',  # TODO: Получать реальные данные
                'change_24h': 0,  # TODO: Рассчитывать реальное изменение
                'momentum': momentum,
            })
        
        # Получить статистику мониторинга
        monitor_stats = scanner.position_monitor.get_stats()
        
        # Получить количество кандидатов для навигации (из БД)
        try:
            session = db_manager.get_session()
            candidates_count = session.query(MarketCandidate).filter(
                MarketCandidate.scan_id == scanner.market_scanner.last_scan_id
            ).count() if scanner.market_scanner.last_scan_id > 0 else 0
            session.close()
        except:
            candidates_count = 0
        
        return render_template('scanner.html',
                             positions=positions,
                             active_count=monitor_stats['active_positions'],
                             total_auto_closes=monitor_stats['total_auto_closes'],
                             success_rate=round(monitor_stats['success_rate'], 1),
                             last_update=monitor_stats['last_monitor_time'].strftime('%H:%M:%S') if monitor_stats['last_monitor_time'] else 'Never',
                             update_time=datetime.now().strftime('%H:%M:%S'),
                             # Для навигации ⭐
                             active_positions_count=monitor_stats['active_positions'],
                             candidates_count=candidates_count)
        
    except Exception as e:
        logger.error(f"Error rendering scanner page: {e}", exc_info=True)
        return render_template('scanner.html',
                             positions=[],
                             active_count=0,
                             total_auto_closes=0,
                             success_rate=0,
                             last_update='Error',
                             update_time=datetime.now().strftime('%H:%M:%S'))

@app.route('/report', methods=['GET'])
def report_route():
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/candidates', methods=['GET'])
def candidates_page():
    """Candidates Page - Market scanning results"""
    try:
        # Получить фильтры из параметров
        filter_direction = request.args.get('direction', 'all')
        filter_strength = request.args.get('strength', 'all')
        min_rating = int(request.args.get('rating', 0))
        
        # Получить кандидатов из MarketScanner
        all_candidates = scanner.market_scanner.get_latest_candidates(limit=50)
        
        # Применить фильтры
        filtered_candidates = []
        for candidate in all_candidates:
            # Direction filter
            if filter_direction != 'all':
                if candidate['direction'].lower() != filter_direction:
                    continue
            
            # Strength filter
            if filter_strength != 'all':
                if candidate['signal_strength'].lower() != filter_strength:
                    continue
            
            # Rating filter
            if candidate['rating'] < min_rating:
                continue
            
            filtered_candidates.append(candidate)
        
        # Подготовить данные для UI
        candidates_ui = []
        for i, candidate in enumerate(filtered_candidates, 1):
            # Определить цвет RSI
            rsi = candidate['rsi']
            if rsi < 30:
                rsi_color = 'rsi-green'
                rsi_text_color = 'var(--green)'
            elif rsi > 70:
                rsi_color = 'rsi-red'
                rsi_text_color = 'var(--red)'
            else:
                rsi_color = 'rsi-yellow'
                rsi_text_color = 'var(--yellow)'
            
            # MFI trend class
            mfi_trend = candidate['mfi_trend']
            if 'Бычий' in mfi_trend or 'Bullish' in mfi_trend:
                mfi_trend_class = 'bullish'
            elif 'Медвежий' in mfi_trend or 'Bearish' in mfi_trend:
                mfi_trend_class = 'bearish'
            else:
                mfi_trend_class = 'neutral'
            
            # Форматировать объём
            volume = candidate['volume_24h']
            if volume >= 1_000_000_000:
                volume_formatted = f"${volume/1_000_000_000:.2f}B"
            elif volume >= 1_000_000:
                volume_formatted = f"${volume/1_000_000:.2f}M"
            else:
                volume_formatted = f"${volume/1_000:.2f}K"
            
            # Рассчитать предлагаемую стратегию
            price = candidate['price']
            direction = candidate['direction']
            
            if direction == 'Long':
                entry = price
                stop_loss = price * 0.97  # -3%
                take_profit = price * 1.05  # +5%
            else:  # Short
                entry = price
                stop_loss = price * 1.03  # +3%
                take_profit = price * 0.95  # -5%
            
            sl_percent = ((stop_loss - entry) / entry) * 100
            tp_percent = ((take_profit - entry) / entry) * 100
            rr_ratio = abs(tp_percent) / abs(sl_percent)
            
            candidates_ui.append({
                'rank': i,
                'symbol': candidate['symbol'],
                'direction': candidate['direction'],
                'signal_strength': candidate['signal_strength'],
                'rating': candidate['rating'],
                'price': candidate['price'],
                'volume_24h': candidate['volume_24h'],
                'volume_24h_formatted': volume_formatted,
                'change_24h': candidate['change_24h'],
                'rsi': candidate['rsi'],
                'rsi_color': rsi_color,
                'rsi_text_color': rsi_text_color,
                'mfi': candidate['mfi'],
                'mfi_trend': candidate['mfi_trend'],
                'mfi_trend_class': mfi_trend_class,
                'reason': candidate['reason'],
                'strategy': {
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'sl_percent': sl_percent,
                    'tp_percent': tp_percent,
                    'rr_ratio': rr_ratio,
                }
            })
        
        # Статистика сканирования
        scanner_stats = scanner.market_scanner.get_stats()
        
        # Время до следующего сканирования
        scanner_params = scanner_config.get_scanner_params()
        scan_interval = scanner_params['scan_interval']
        
        if scanner_stats['last_scan_time']:
            elapsed = (datetime.now() - scanner_stats['last_scan_time']).total_seconds()
            next_scan_in = max(0, int(scan_interval - elapsed))
        else:
            next_scan_in = scan_interval
        
        # Получить количество активных позиций для навигации
        monitor_stats = scanner.position_monitor.get_stats()
        active_positions_count = monitor_stats['active_positions']
        
        return render_template('candidates.html',
                             candidates=candidates_ui,
                             candidates_count=len(all_candidates),
                             total_scans=scanner_stats['total_scans'],
                             scan_duration='N/A',  # TODO: получать из БД
                             last_scan_time=scanner_stats['last_scan_time'].strftime('%H:%M:%S') if scanner_stats['last_scan_time'] else 'Never',
                             update_time=datetime.now().strftime('%H:%M:%S'),
                             filter_direction=filter_direction,
                             filter_strength=filter_strength,
                             min_rating=min_rating,
                             scanning=scanner_stats['scanning'],
                             next_scan_in=next_scan_in,
                             # Для навигации ⭐
                             active_positions_count=active_positions_count)
        
    except Exception as e:
        logger.error(f"Error rendering candidates page: {e}", exc_info=True)
        return render_template('candidates.html',
                             candidates=[],
                             candidates_count=0,
                             total_scans=0,
                             scan_duration='Error',
                             last_scan_time='Error',
                             update_time=datetime.now().strftime('%H:%M:%S'),
                             filter_direction='all',
                             filter_strength='all',
                             min_rating=0,
                             scanning=False,
                             next_scan_in=60)

@app.route('/api/scan', methods=['POST'])
def api_scan():
    """API endpoint для ручного запуска сканирования"""
    try:
        # Запустить сканирование
        results = scanner.market_scanner.scan_market()
        
        return jsonify({
            'status': 'ok',
            'candidates_count': len(results),
            'message': f'Scan completed, found {len(results)} candidates'
        })
    except Exception as e:
        logger.error(f"API scan error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/parameters', methods=['GET', 'POST'])
def parameters_page():
    """Parameters Page - Configuration management"""
    message = None
    message_type = None
    
    try:
        # Handle preset application
        if request.method == 'GET':
            preset = request.args.get('preset')
            reset = request.args.get('reset')
            
            if preset:
                if preset == 'scalping':
                    scanner_config.update_trading_style('scalping')
                    message = 'Scalping preset applied!'
                    message_type = 'success'
                elif preset == 'daytrading':
                    scanner_config.update_trading_style('daytrading')
                    message = 'Day Trading preset applied!'
                    message_type = 'success'
                elif preset == 'swing':
                    scanner_config.update_trading_style('swing')
                    message = 'Swing Trading preset applied!'
                    message_type = 'success'
            
            elif reset:
                # Reset to defaults
                scanner_config.trading_style = 'daytrading'
                scanner_config.aggressiveness = 'auto'
                scanner_config.automation_mode = 'semi_auto'
                scanner_config.update_trading_style('daytrading')
                scanner_config.update_aggressiveness('auto')
                scanner_config.update_automation('semi_auto')
                message = 'Parameters reset to defaults!'
                message_type = 'success'
        
        # Handle form submission
        if request.method == 'POST':
            try:
                # Функції безпечної конвертації ⭐ НОВЕ v2.6.2
                def safe_int(value, default):
                    try:
                        if value == '' or value is None:
                            return default
                        return int(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid int value: {value}, using default: {default}")
                        return default
                
                def safe_float(value, default):
                    try:
                        if value == '' or value is None:
                            return default
                        return float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid float value: {value}, using default: {default}")
                        return default
                
                # Trading Style
                trading_style = request.form.get('trading_style')
                aggressiveness = request.form.get('aggressiveness')
                automation_mode = request.form.get('automation_mode')
                
                scanner_config.update_trading_style(trading_style)
                scanner_config.update_aggressiveness(aggressiveness)
                scanner_config.update_automation(automation_mode)
                
                # Timeframe (ВАЖНО!)
                timeframe = request.form.get('indicator_timeframe', '240')
                scanner_config.set_timeframe(timeframe)
                
                # Keep-Alive (НОВОЕ) ⭐
                scanner_config.keep_alive_enabled = 'keep_alive_enabled' in request.form
                scanner_config.keep_alive_interval = safe_int(request.form.get('keep_alive_interval'), 300)
                
                # Indicator Parameters
                scanner_config.update_param('indicator', 'rsi_period', safe_int(request.form.get('rsi_period', 14)))
                scanner_config.update_param('indicator', 'rsi_oversold', safe_int(request.form.get('rsi_oversold', 30)))
                scanner_config.update_param('indicator', 'rsi_overbought', safe_int(request.form.get('rsi_overbought', 70)))
                scanner_config.update_param('indicator', 'mfi_period', safe_int(request.form.get('mfi_period', 20)))
                scanner_config.update_param('indicator', 'mfi_fast_ema', safe_int(request.form.get('mfi_fast_ema', 5)))
                scanner_config.update_param('indicator', 'mfi_slow_ema', safe_int(request.form.get('mfi_slow_ema', 13)))
                
                # Risk Management
                scanner_config.update_param('risk', 'max_positions', safe_int(request.form.get('max_positions', 3)))
                scanner_config.update_param('risk', 'position_size_percent', safe_float(request.form.get('position_size_percent', 10)))
                # Daily Loss Limit: конвертуємо позитивне значення в негативне
                daily_loss = safe_float(request.form.get('daily_loss_limit_percent', 5))
                daily_loss = -abs(daily_loss)  # Завжди негативне
                scanner_config.update_param('risk', 'daily_loss_limit_percent', daily_loss)
                scanner_config.update_param('risk', 'default_leverage', safe_int(request.form.get('default_leverage', 20)))
                scanner_config.update_param('risk', 'reserve_balance', safe_float(request.form.get('reserve_balance', 100)))
                
                # Auto-Close
                scanner_config.update_param('auto_close', 'enabled', 'auto_close_enabled' in request.form)
                scanner_config.update_param('auto_close', 'use_strong_signals', 'use_strong_signals' in request.form)
                scanner_config.update_param('auto_close', 'confirm_with_mfi', 'confirm_with_mfi' in request.form)
                scanner_config.update_param('auto_close', 'min_hold_time', safe_int(request.form.get('min_hold_time', 300)))
                
                # OBV Parameters (НОВОЕ) ⭐
                scanner_config.update_param('auto_close', 'use_obv_confirmation', 'use_obv_confirmation' in request.form)
                scanner_config.update_param('auto_close', 'obv_ema_period', safe_int(request.form.get('obv_ema_period', 20)))
                scanner_config.update_param('auto_close', 'obv_trend_candles', safe_int(request.form.get('obv_trend_candles', 3)))
                scanner_config.update_param('auto_close', 'obv_sensitivity', request.form.get('obv_sensitivity', 'high'))
                scanner_config.update_param('auto_close', 'rsi_exit_mode', request.form.get('rsi_exit_mode', 'wait_zone_exit'))
                scanner_config.update_param('auto_close', 'mfi_check_mode', request.form.get('mfi_check_mode', 'after_obv'))
                
                # ⭐ НОВЕ v2.4: TP Strategy
                tp_strategy = request.form.get('tp_strategy', 'balanced')
                try:
                    from tp_strategy_config import tp_config
                    tp_config.set_strategy(tp_strategy)
                    logger.info(f"✅ TP Strategy set to: {tp_strategy}")
                except Exception as e:
                    logger.error(f"❌ Error setting TP strategy: {e}")
                
                # Scanner
                scanner_config.update_param('scanner', 'enabled', 'scanner_enabled' in request.form)
                scanner_config.update_param('scanner', 'scan_interval', int(request.form.get('scan_interval', 60)))
                scanner_config.update_param('scanner', 'top_candidates_count', int(request.form.get('top_candidates_count', 10)))
                scanner_config.update_param('scanner', 'min_volume_24h', float(request.form.get('min_volume_24h', 10)) * 1_000_000)
                scanner_config.update_param('scanner', 'min_price_change_24h', float(request.form.get('min_price_change_24h', 2)))
                scanner_config.update_param('scanner', 'min_signal_strength', request.form.get('min_signal_strength', 'regular'))
                
                # Save to file
                scanner_config.save_to_json('scanner_config.json')
                
                message = 'All parameters saved successfully!'
                message_type = 'success'
                
                logger.info("✅ Параметри успішно оновлено")
                
            except Exception as e:
                message = f'Error saving parameters: {str(e)}'
                message_type = 'error'
                logger.error(f"Error saving parameters: {e}")
        
        # Get current parameters
        timeframe = scanner_config.get_timeframe()
        timeframe_labels = {
            '1': '1 Minute',
            '5': '5 Minutes',
            '15': '15 Minutes',
            '60': '1 Hour',
            '240': '4 Hours',
            'D': '1 Day',
        }
        
        params = {
            'trading_style': scanner_config.trading_style,
            'aggressiveness': scanner_config.aggressiveness,
            'automation_mode': scanner_config.automation_mode,
            'indicator_timeframe': timeframe,
            'indicator_timeframe_label': timeframe_labels.get(timeframe, '4 Hours'),
            'keep_alive_enabled': scanner_config.keep_alive_enabled,  # ⭐ НОВОЕ
            'keep_alive_interval': scanner_config.keep_alive_interval,  # ⭐ НОВОЕ
            'indicator': scanner_config.get_indicator_params(),
            'risk': scanner_config.get_risk_params(),
            'auto_close': scanner_config.get_auto_close_params(),
            'scanner': scanner_config.get_scanner_params(),
        }
        
        # Получить счётчики для навигации
        monitor_stats = scanner.position_monitor.get_stats()
        
        # Получить количество кандидатов из БД
        try:
            session = db_manager.get_session()
            candidates_count = session.query(MarketCandidate).filter(
                MarketCandidate.scan_id == scanner.market_scanner.last_scan_id
            ).count() if scanner.market_scanner.last_scan_id > 0 else 0
            session.close()
        except:
            candidates_count = 0
        
        # Отримати поточну TP стратегію
        try:
            from tp_strategy_config import tp_config
            current_tp_strategy = tp_config.current_strategy
        except:
            current_tp_strategy = 'balanced'
        
        return render_template('parameters.html',
                             params=params,
                             current_style=scanner_config.trading_style,
                             current_tp_strategy=current_tp_strategy,  # ⭐ НОВЕ
                             message=message,
                             message_type=message_type,
                             # Для навигации ⭐
                             active_positions_count=monitor_stats['active_positions'],
                             candidates_count=candidates_count)
        
    except Exception as e:
        logger.error(f"Error rendering parameters page: {e}", exc_info=True)
        return render_template('parameters.html',
                             params={},
                             current_style='daytrading',
                             message='Error loading parameters',
                             message_type='error')

@app.route('/api/config/export', methods=['GET'])
def api_config_export():
    """Export configuration as JSON"""
    try:
        config_dict = scanner_config.to_dict()
        return jsonify(config_dict)
    except Exception as e:
        logger.error(f"Config export error: {e}")
        return jsonify({'error': str(e)}), 500

# ⭐ API для налаштувань сканера v2.3
@app.route('/api/scanner/settings', methods=['GET', 'POST'])
def scanner_settings_api():
    """Отримати або зберегти налаштування сканера"""
    if request.method == 'GET':
        try:
            settings = scanner_config.get_scanner_params()
            return jsonify({'status': 'ok', 'settings': settings})
        except Exception as e:
            logger.error(f"Помилка отримання налаштувань: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    else:  # POST
        try:
            data = request.json
            
            # Оновити параметри сканера
            scanner_config.update_scanner_param('timeframe', data.get('scanner_timeframe', '240'))
            scanner_config.update_scanner_param('rsi_oversold', int(data.get('scanner_rsi_oversold', 45)))
            scanner_config.update_scanner_param('rsi_overbought', int(data.get('scanner_rsi_overbought', 55)))
            scanner_config.update_scanner_param('rsi_period', int(data.get('scanner_rsi_period', 14)))
            
            scanner_config.update_scanner_param('min_volume_24h', int(data.get('scanner_min_volume', 3000000)))
            scanner_config.update_scanner_param('min_price_change_24h', float(data.get('scanner_min_change', 0.8)))
            
            scanner_config.update_scanner_param('require_volume', data.get('scanner_require_volume', False))
            scanner_config.update_scanner_param('trend_confirmation', data.get('scanner_trend_confirmation', False))
            scanner_config.update_scanner_param('top_candidates_count', int(data.get('scanner_top_count', 10)))
            scanner_config.update_scanner_param('batch_size', int(data.get('scanner_batch_size', 30)))
            
            logger.info("✅ Налаштування сканера оновлено")
            return jsonify({'status': 'ok', 'message': 'Налаштування збережено'})
        except Exception as e:
            logger.error(f"Помилка збереження налаштувань сканера: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/scanner/settings/reset', methods=['POST'])
def reset_scanner_settings_api():
    """Скинути налаштування сканера до типових"""
    try:
        scanner_config.scanner_params = scanner_config._get_default_scanner_params()
        logger.info("✅ Налаштування сканера скинуто до типових")
        return jsonify({'status': 'ok', 'message': 'Налаштування скинуто до типових'})
    except Exception as e:
        logger.error(f"Помилка скидання налаштувань: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
