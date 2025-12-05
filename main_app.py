#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlgoBot Trading Application
Flask app з webhook поддержкой, CSRF захистом і структурованим логуванням
"""
import threading
import time
import json
import ctypes
import os
import requests
import pandas as pd
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, session
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import desc

from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from settings_manager import settings
from models import db_manager, OrderBlock, PaperTrade, SmartMoneyTicker
from market_analyzer import market_analyzer
from config import get_api_credentials
from utils import get_logger, validate_webhook_data, metrics, setup_logging

# === ІНІЦІАЛІЗАЦІЯ ЛОГУВАННЯ ===
setup_logging()
logger = get_logger()

app = Flask(__name__)

# 🔐 SECRET KEY ЗІ ЗМІННИХ (КРИТИЧНЕ!)
secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    if os.environ.get('RENDER'):
        raise ValueError("❌ FLASK_SECRET_KEY not set! This is REQUIRED for security on Render.")
    else:
        # Локальна розробка - генеруємо
        import secrets
        secret_key = secrets.token_hex(16)
        print(f"⚠️ FLASK_SECRET_KEY not found, using random for development: {secret_key}")

app.config['SECRET_KEY'] = secret_key
app.config['WTF_CSRF_TIME_LIMIT'] = None  # Без часових обмежень для CSRF токена
app.config['WTF_CSRF_CHECK_DEFAULT'] = True

csrf = CSRFProtect(app)

# === СИСТЕМНА УТИЛІТА (Windows) ===
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except:
    pass

logger.info("app_initialized", env=os.environ.get('FLASK_ENV', 'development'))

# === ІНІЦІАЛІЗАЦІЯ СКАНЕРА ===
try:
    scanner = EnhancedMarketScanner(bot_instance, {})
    scanner.start()
    logger.info("scanner_started")
except Exception as e:
    logger.error("scanner_init_failed", error=str(e), exc_info=True)

# ===== ФОНОВІ ПОТОКИ =====

def monitor_active():
    """🔄 Мониторит активные позиции з логуванням"""
    logger.info("monitor_active_started")
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r.get('retCode') == 0:
                for p in r['result']['list']:
                    if float(p.get('size', 0)) > 0:
                        try:
                            stats_service.save_monitor_log({
                                'symbol': p['symbol'],
                                'price': float(p.get('avgPrice', 0)),
                                'pnl': float(p.get('unrealisedPnl', 0)),
                                'rsi': scanner.get_current_rsi(p['symbol']),
                                'pressure': scanner.get_market_pressure(p['symbol'])
                            })
                        except Exception as e:
                            logger.warning("monitor_save_failed", symbol=p.get('symbol'), error=str(e))
        except Exception as e:
            logger.error("monitor_active_error", error=str(e))
        
        time.sleep(10)

def keep_alive():
    """🫀 Keep-alive для хостів (Render, Heroku)"""
    time.sleep(5)
    base_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not base_url:
        port = os.environ.get('PORT', 10000)
        base_url = f'http://127.0.0.1:{port}'
    
    target = f"{base_url}/health"
    logger.info("keep_alive_started", target=target)
    
    while True:
        try:
            requests.get(target, timeout=10)
        except Exception as e:
            logger.warning("keep_alive_failed", error=str(e))
        time.sleep(300)

def sync_trades_periodic():
    """📊 Синхронізує торги кожні 30 хвилин"""
    time.sleep(5)
    sync_interval = int(os.environ.get('TRADES_SYNC_INTERVAL', 1800))
    logger.info("sync_trades_started", interval_sec=sync_interval)
    
    while True:
        try:
            bot_instance.sync_trades(days=7)
            logger.info("periodic_sync_completed")
        except Exception as e:
            logger.error("periodic_sync_error", error=str(e), exc_info=True)
        
        time.sleep(sync_interval)

# Запускаємо потоки
threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()
threading.Thread(target=sync_trades_periodic, daemon=True).start()

logger.info("background_threads_started", count=3)

# ===== MIDDLEWARE =====

@app.before_request
def before_request():
    """Логування кожного запиту"""
    request.start_time = time.time()
    logger.info("request_received", 
               method=request.method,
               path=request.path,
               remote_addr=request.remote_addr)

@app.after_request
def after_request(response):
    """Логування відповіді"""
    duration = time.time() - request.start_time
    logger.info("request_completed",
               method=request.method,
               path=request.path,
               status=response.status_code,
               duration_ms=round(duration * 1000, 2))
    return response

@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def error_handler(error):
    """Обробка помилок"""
    logger.error("request_error", 
                status=error.code,
                message=str(error),
                path=request.path)
    return jsonify({"error": str(error)}), error.code

# ===== API МАРШУТИ =====

@app.route('/health', methods=['GET'])
@csrf.exempt  # Health check не потребує CSRF
def health():
    """Перевірка здоров'я додатку"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics.get_stats()
    }), 200

@app.route('/api/chart_data/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """Отримує дані для графіку"""
    try:
        clean_symbol = symbol.replace('.P', '')
        htf = settings.get("htfSelection", "240")
        df = market_analyzer.fetch_candles(clean_symbol, htf, limit=200)
        
        candles = []
        if df is not None:
            for _, row in df.iterrows():
                candles.append({
                    'time': int(row['time'].timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
        
        # Отримуємо блоки ордерів
        session = db_manager.get_session()
        try:
            db_blocks = session.query(OrderBlock).filter(
                OrderBlock.symbol.in_([symbol, clean_symbol])
            ).all()
            blocks = [
                {'type': b.ob_type, 'top': b.top, 'bottom': b.bottom, 'timeframe': b.timeframe}
                for b in db_blocks
            ]
        finally:
            session.close()
        
        return jsonify({'candles': candles, 'blocks': blocks})
    except Exception as e:
        logger.error("chart_data_error", symbol=symbol, error=str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """Головна сторінка з статистикою"""
    try:
        days_param = int(request.args.get('days', 7))
    except:
        days_param = 7
    
    if days_param not in [7, 30, 90]:
        days_param = 7
    
    try:
        bot_instance.sync_trades(days=days_param)
    except Exception as e:
        logger.warning("home_sync_failed", error=str(e))
    
    balance = bot_instance.get_bal()
    active_count = len(scanner.get_active_symbols())
    trades = stats_service.get_trades(days=days_param)
    period_pnl = sum(t['pnl'] for t in trades) if trades else 0
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
                          trades=trades[:10] if trades else [])

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Монітор активних торгів"""
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r.get('retCode') == 0:
            for p in r['result']['list']:
                if float(p.get('size', 0)) > 0:
                    symbol = p['symbol']
                    coin_data = scanner.get_coin_data(symbol)
                    active.append({
                        'symbol': symbol,
                        'side': p['side'],
                        'pnl': round(float(p.get('unrealisedPnl', 0)), 2),
                        'rsi': coin_data.get('rsi', 0),
                        'exit_status': coin_data.get('exit_status', 'Safe'),
                        'exit_details': coin_data.get('exit_details', '-'),
                        'pressure': round(scanner.get_market_pressure(symbol), 1),
                        'size': p.get('size', 0),
                        'entry': p.get('avgPrice', 0),
                        'time': datetime.now().strftime('%H:%M')
                    })
    except Exception as e:
        logger.error("scanner_page_error", error=str(e), exc_info=True)
    
    return render_template('scanner.html', active=active, conf=settings._cache)

@app.route('/analyzer')
def analyzer_page():
    """Сканер ринку"""
    results = market_analyzer.get_results()
    conf = settings._cache
    return render_template('analyzer.html',
                          results=results,
                          conf=conf,
                          progress=market_analyzer.progress,
                          status=market_analyzer.status_message,
                          is_scanning=market_analyzer.is_scanning)

@app.route('/smart_money', methods=['GET', 'POST'])
def smart_money_page():
    """Smart Money вотчліст"""
    if request.method == 'POST':
        form_data = request.form.to_dict()
        settings.save_settings(form_data)
        return redirect(url_for('smart_money_page'))
    
    session_db = db_manager.get_session()
    try:
        watchlist = session_db.query(SmartMoneyTicker).all()
        active_trades = session_db.query(PaperTrade).filter(
            PaperTrade.status.in_(['OPEN', 'PENDING'])
        ).order_by(desc(PaperTrade.created_at)).all()
        history_trades = session_db.query(PaperTrade).filter(
            PaperTrade.status.in_(['CLOSED_WIN', 'CLOSED_LOSS', 'CANCELED'])
        ).order_by(desc(PaperTrade.closed_at)).limit(50).all()
        
        return render_template('smart_money.html',
                              watchlist=watchlist,
                              active_trades=active_trades,
                              history_trades=history_trades,
                              conf=settings._cache)
    finally:
        session_db.close()

@app.route('/smart_money/delete/<symbol>', methods=['POST'])
def delete_ticker(symbol):
    """Видалення из вотчліста"""
    session_db = db_manager.get_session()
    try:
        item = session_db.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
        if item:
            session_db.delete(item)
            session_db.commit()
            logger.info("ticker_deleted", symbol=symbol)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error("delete_ticker_error", symbol=symbol, error=str(e))
        return jsonify({'error': str(e)}), 400
    finally:
        session_db.close()

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    """Загальні налаштування"""
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        form_data['exit_enableStrategy'] = request.form.get('exit_enableStrategy') == 'on'
        settings.save_settings(form_data)
        logger.info("settings_saved", user_agent=request.user_agent)
        return redirect(url_for('settings_general_page'))
    
    return render_template('settings.html', conf=settings._cache)

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def ob_trend_settings_page():
    """Налаштування стратегії OB Trend"""
    if request.method == 'POST':
        form_data = request.form.to_dict()
        filters = ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter', 'obt_useOBRetest']
        for cb in filters:
            form_data[cb] = request.form.get(cb) == 'on'
        settings.save_settings(form_data)
        logger.info("ob_trend_settings_saved")
        return redirect(url_for('ob_trend_settings_page'))
    
    return render_template('strategy_ob_trend.html', conf=settings._cache)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    """Запускає сканер ринку"""
    try:
        if request.form:
            form_data = request.form.to_dict()
            checkboxes = ['obt_useOBRetest', 'obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter']
            for cb in checkboxes:
                form_data[cb] = request.form.get(cb) == 'on'
            settings.save_settings(form_data)
        
        market_analyzer.run_scan_thread()
        logger.info("scan_started")
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error("run_scan_error", error=str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/analyzer/status')
def get_scan_status():
    """Отримує статус сканування"""
    return jsonify({
        "progress": market_analyzer.progress,
        "message": market_analyzer.status_message,
        "is_scanning": market_analyzer.is_scanning
    })

@app.route('/webhook', methods=['POST'])
@csrf.exempt  # Webhook від TradingView не має CSRF токена
def webhook():
    """
    Webhook для приймання сигналів з TradingView
    
    Очікує JSON:
    {
        "action": "Buy|Sell|Close",
        "symbol": "BTCUSDT",
        "direction": "Long|Short" (для Close),
        "riskPercent": 2.0,
        "leverage": 20,
        "sl_price": float (опціонально),
        "tp_price": float (опціонально)
    }
    """
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info("webhook_received", action=data.get('action'), symbol=data.get('symbol'))
        
        # Валідуємо дані (буде викине ValueError якщо неправильно)
        result = bot_instance.place_order(data)
        
        status_code = 200 if result.get("status") in ["ok", "ignored"] else 400
        logger.info("webhook_processed", status=result.get("status"), code=status_code)
        
        return jsonify(result), status_code
    
    except ValueError as e:
        # Помилка валідації
        logger.warning("webhook_validation_error", error=str(e))
        return jsonify({"error": f"Invalid webhook data: {str(e)}", "code": "VALIDATION_ERROR"}), 400
    except Exception as e:
        # Неочікувана помилка
        logger.error("webhook_error", error=str(e), exc_info=True)
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500

@app.route('/settings/export')
def export_settings():
    """Експортує налаштування у JSON"""
    try:
        json_str = json.dumps(settings.get_all(), indent=4)
        logger.info("settings_exported")
        return Response(json_str,
                       mimetype='application/json',
                       headers={'Content-Disposition': 'attachment;filename=bot_settings.json'})
    except Exception as e:
        logger.error("export_settings_error", error=str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/settings/import', methods=['POST'])
def import_settings():
    """Імпортує налаштування з JSON файлу"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if settings.import_settings(json.load(file)):
            logger.info("settings_imported")
            return redirect(url_for('settings_general_page'))
        else:
            logger.warning("settings_import_failed")
            return jsonify({"error": "Failed to import settings"}), 400
    except Exception as e:
        logger.error("import_settings_error", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===== ЗАПУСК =====

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info("starting_flask", host=host, port=port, debug=debug)
    app.run(host=host, port=port, debug=debug)

