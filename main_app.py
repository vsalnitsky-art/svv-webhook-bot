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
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, session, g
from flask_wtf.csrf import CSRFProtect, generate_csrf
from sqlalchemy import desc

from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from settings_manager import settings
from models import db_manager, OrderBlock, WhaleSignal, SmartMoneyTicker, DetectedOrderBlock
from market_analyzer import market_analyzer
from config import get_api_credentials
from utils import get_logger, validate_webhook_data, metrics, setup_logging

# === WHALE MODULE IMPORT (INTEGRATION) ===
from whale_core import whale_core
# ✅ IMPORT WHALE PRO
from whale_pro import register_routes as register_whale_pro
# ✅ IMPORT WHALE SNIPER (NEW STRATEGY)
from sniper_strategy import sniper_bot

# === ІНІЦІАЛІЗАЦІЯ ЛОГУВАННЯ ===
setup_logging()

# === SAFE FLOAT HELPER ===
def safe_float(val, default=0.0):
    """
    Безпечне конвертування в float.
    Bybit API може повертати '', None, або невалідні значення.
    """
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
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

# === CSRF CONTEXT PROCESSOR ===
@app.context_processor
def inject_csrf_token():
    """Автоматично передає csrf_token функцію у всі шаблони"""
    return dict(csrf_token=generate_csrf)

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
                    size = safe_float(p.get('size'), 0)
                    if size > 0:
                        try:
                            stats_service.save_monitor_log({
                                'symbol': p['symbol'],
                                'price': safe_float(p.get('avgPrice'), 0),
                                'pnl': safe_float(p.get('unrealisedPnl'), 0),
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
    g.start_time = time.time()
    logger.info("request_received", 
               method=request.method,
               path=request.path,
               remote_addr=request.remote_addr)

@app.after_request
def after_request(response):
    """Логування відповіді"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
    else:
        duration = 0.0
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

# ===== ДОПОМІЖНІ ФУНКЦІЇ СТАТИСТИКИ (ПОВНА ВЕРСІЯ) =====

def calculate_stats(trades):
    """
    Розраховує детальну статистику торгів
    Повертає словник зі статистикою та рейтингом контрактів
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
    
    # Розрахунок стриків
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
    contract_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'volume': 0})
    
    for t in trades:
        symbol = t.get('symbol', 'Unknown')
        contract_stats[symbol]['trades'] += 1
        if t.get('pnl', 0) > 0:
            contract_stats[symbol]['wins'] += 1
        contract_stats[symbol]['pnl'] += t.get('pnl', 0)
        contract_stats[symbol]['volume'] += t.get('qty', 0)
    
    # Додати win_rate для кожного контракту
    for symbol in contract_stats:
        stats = contract_stats[symbol]
        stats['win_rate'] = round(stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0, 1)
        stats['avg_pnl'] = round(stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0, 2)
    
    # Сортувати по P&L (спадаючи) - топ 10
    sorted_contracts = sorted(
        contract_stats.items(),
        key=lambda x: x[1]['pnl'],
        reverse=True
    )[:10]
    
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

# ==========================================
# 🐋 WHALE STRATEGY MODULE ROUTES
# ==========================================
@app.route('/whale')
def whale_page():
    # Отримуємо історію з БД через метод ядра
    history = whale_core.get_history(limit=50)
    
    return render_template(
        'whale.html',
        history=history,
        is_scanning=whale_core.is_scanning,
        progress=whale_core.progress,
        status=whale_core.status,
        last_time=whale_core.last_scan_time,
        conf=settings._cache # Для сумісності з base.html
    )

@app.route('/whale/scan', methods=['POST'])
def whale_scan_start():
    data = request.json or {}
    started = whale_core.start_scan(override_cfg=data)
    return jsonify({"status": "started" if started else "busy"})

# ==========================================
# 🎯 WHALE SNIPER (NEW!)
# ==========================================
@app.route('/sniper')
def sniper_page():
    """Page for the new Sniper Strategy"""
    return render_template('sniper.html',
                         status=sniper_bot.status,
                         is_running=sniper_bot.is_running,
                         history=sniper_bot.found_signals)

@app.route('/sniper/toggle', methods=['POST'])
def sniper_toggle():
    """Start/Stop Sniper"""
    data = request.json or {}
    enable = data.get('enable', False)
    
    if enable:
        sniper_bot.start()
    else:
        sniper_bot.stop()
        
    return jsonify({'status': 'ok', 'is_running': sniper_bot.is_running})

# ==========================================

@app.route('/', methods=['GET'])
def index_page():
    """ПРОФЕСІЙНИЙ ОГЛЯД РИНКУ з детальною статистикою"""
    try:
        days_param = int(request.args.get('days', 7))
    except:
        days_param = 7
    
    # Дозволені періоди
    if days_param not in [7, 30, 60, 90, 180]:
        days_param = 7
    
    try:
        # Синхронізуємо торги для обраного періоду
        bot_instance.sync_trades(days=days_param)
    except Exception as e:
        logger.warning("index_sync_failed", error=str(e))
    
    # Отримуємо баланс
    try:
        balance = bot_instance.get_available_balance()
    except:
        balance = 0
    
    # Отримуємо активні позиції
    try:
        active_positions = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if active_positions.get('retCode') == 0:
            active_count = len([p for p in active_positions['result']['list'] if safe_float(p.get('size'), 0) > 0])
        else:
            active_count = 0
    except:
        active_count = 0
    
    # Отримуємо торги за період
    trades = stats_service.get_trades(days=days_param)
    
    # Розраховуємо детальну статистику
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
                          stats=stats,
                          conf=settings._cache)

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Монітор активних торгів"""
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r.get('retCode') == 0:
            for p in r['result']['list']:
                size = safe_float(p.get('size'), 0)
                if size > 0:
                    symbol = p['symbol']
                    coin_data = scanner.get_coin_data(symbol)
                    
                    # ✨ ФОРМАТУВАННЯ ЧИСЕЛ
                    entry = safe_float(p.get('avgPrice'), 0)
                    
                    # Визначаємо скільки знаків для ціни (залежить від вартості)
                    if entry >= 1000:
                        entry_rounded = round(entry, 1)
                    elif entry >= 1:
                        entry_rounded = round(entry, 2)
                    else:
                        entry_rounded = round(entry, 6)
                    
                    # Обсяг - максимум 3 знаки після коми
                    size_rounded = round(size, 3) if size < 100 else round(size, 2)
                    
                    active.append({
                        'symbol': symbol,
                        'side': p['side'],
                        'pnl': round(safe_float(p.get('unrealisedPnl'), 0), 2),
                        'rsi': coin_data.get('rsi', 0),
                        'exit_status': coin_data.get('exit_status', 'Safe'),
                        'exit_details': coin_data.get('exit_details', '-'),
                        'pressure': round(scanner.get_market_pressure(symbol), 1),
                        'size': size_rounded,
                        'entry': entry_rounded,
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

@app.route('/smart_money')
def smart_money_page():
    """Smart Money - Order Block Scanner"""
    return render_template('smart_money.html', conf=settings._cache)


@app.route('/smart_money/settings', methods=['GET', 'POST'])
def smart_money_settings():
    """Налаштування Smart Money"""
    if request.method == 'POST':
        data = request.get_json() or {}
        logger.info(f"📥 POST /smart_money/settings: {data}")
        settings.save_settings(data)
        
        # Оновити scheduler якщо змінилось auto_scan
        update_ob_scheduler()
        
        return jsonify({'status': 'ok'})
    
    # GET - перезавантажуємо з БД для актуальності (для multi-worker)
    settings.reload_settings()
    
    # Явно конвертуємо boolean значення
    auto_scan = settings.get('ob_auto_scan', False)
    auto_add = settings.get('ob_auto_add_from_screener', False)
    exec_trades = settings.get('ob_execute_trades', False)
    
    # Ensure they are actual booleans
    auto_scan = bool(auto_scan) if not isinstance(auto_scan, bool) else auto_scan
    auto_add = bool(auto_add) if not isinstance(auto_add, bool) else auto_add
    exec_trades = bool(exec_trades) if not isinstance(exec_trades, bool) else exec_trades
    
    logger.info(f"📤 GET /smart_money/settings: auto_scan={auto_scan} ({type(auto_scan).__name__}), auto_add={auto_add} ({type(auto_add).__name__})")
    
    result = {
        'ob_source_tf': settings.get('ob_source_tf', '15'),
        'ob_swing_length': settings.get('ob_swing_length', 3),
        'ob_zone_count': settings.get('ob_zone_count', 'High'),
        'ob_max_atr_mult': settings.get('ob_max_atr_mult', 3.5),
        'ob_invalidation_method': settings.get('ob_invalidation_method', 'Wick'),
        'ob_combine_obs': bool(settings.get('ob_combine_obs', True)),
        'ob_entry_mode': settings.get('ob_entry_mode', 'Immediate'),
        'ob_selection': settings.get('ob_selection', 'Newest'),
        'ob_sl_atr_mult': settings.get('ob_sl_atr_mult', 0.3),
        'ob_watchlist_timeout': settings.get('ob_watchlist_timeout', '24h'),
        'ob_scan_interval': settings.get('ob_scan_interval', 60),
        'ob_watchlist_limit': settings.get('ob_watchlist_limit', 50),
        'ob_persistence_check': bool(settings.get('ob_persistence_check', False)),
        'ob_auto_scan': auto_scan,
        'ob_auto_add_from_screener': auto_add,
        'ob_execute_trades': exec_trades
    }
    
    return jsonify(result)


@app.route('/smart_money/watchlist', methods=['GET'])
def smart_money_watchlist():
    """Отримати watchlist"""
    from models import SmartMoneyTicker
    session_db = db_manager.get_session()
    try:
        items = session_db.query(SmartMoneyTicker).order_by(SmartMoneyTicker.added_at.desc()).all()
        return jsonify([{
            'id': item.id,
            'symbol': item.symbol,
            'direction': item.direction or 'BUY',
            'source': item.source or 'Manual',
            'added_at': item.added_at.isoformat() if item.added_at else None
        } for item in items])
    finally:
        session_db.close()


@app.route('/smart_money/watchlist/add', methods=['POST'])
def smart_money_watchlist_add():
    """Додати в watchlist або оновити direction"""
    from models import SmartMoneyTicker
    data = request.get_json() or {}
    symbol = data.get('symbol', '').upper().strip()
    direction = data.get('direction', 'BUY')
    source = data.get('source', 'Manual')
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    if not symbol.endswith('USDT'):
        symbol += 'USDT'
    
    session_db = db_manager.get_session()
    try:
        # Перевірка чи вже існує
        existing = session_db.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
        if existing:
            # Якщо direction відрізняється - оновлюємо
            if existing.direction != direction:
                existing.direction = direction
                existing.source = source
                existing.added_at = datetime.utcnow()
                session_db.commit()
                logger.info(f"Watchlist update: {symbol} direction changed to {direction}")
                return jsonify({'status': 'ok', 'action': 'updated'})
            else:
                return jsonify({'status': 'ok', 'action': 'exists', 'message': 'Symbol already in watchlist with same direction'})
        
        # Перевірка ліміту
        count = session_db.query(SmartMoneyTicker).count()
        limit = settings.get('ob_watchlist_limit', 50)
        if count >= limit:
            return jsonify({'error': f'Watchlist limit reached ({limit})'}), 400
        
        # Додаємо
        new_item = SmartMoneyTicker(symbol=symbol, direction=direction, source=source)
        session_db.add(new_item)
        session_db.commit()
        
        logger.info(f"Watchlist add: {symbol} {direction}")
        return jsonify({'status': 'ok', 'action': 'added'})
    except Exception as e:
        session_db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session_db.close()


@app.route('/smart_money/watchlist/remove/<symbol>', methods=['POST'])
def smart_money_watchlist_remove(symbol):
    """Видалити з watchlist"""
    from models import SmartMoneyTicker
    session_db = db_manager.get_session()
    try:
        item = session_db.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
        if item:
            session_db.delete(item)
            session_db.commit()
            logger.info(f"Watchlist remove: {symbol}")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session_db.close()


@app.route('/smart_money/watchlist/clear', methods=['POST'])
def smart_money_watchlist_clear():
    """Очистити весь watchlist"""
    from models import SmartMoneyTicker
    session_db = db_manager.get_session()
    try:
        count = session_db.query(SmartMoneyTicker).delete()
        session_db.commit()
        logger.info(f"Watchlist cleared: {count} items removed")
        return jsonify({'status': 'ok', 'removed': count})
    except Exception as e:
        session_db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session_db.close()


@app.route('/smart_money/detected', methods=['GET'])
def smart_money_detected():
    """Отримати знайдені Order Blocks"""
    from models import DetectedOrderBlock
    session_db = db_manager.get_session()
    try:
        items = session_db.query(DetectedOrderBlock).order_by(DetectedOrderBlock.detected_at.desc()).limit(100).all()
        return jsonify([{
            'id': item.id,
            'symbol': item.symbol,
            'direction': item.direction,
            'ob_type': item.ob_type,
            'ob_top': item.ob_top,
            'ob_bottom': item.ob_bottom,
            'entry_price': item.entry_price,
            'sl_price': item.sl_price,
            'current_price': item.current_price,
            'status': item.status,
            'timeframe': item.timeframe,
            'detected_at': item.detected_at.isoformat() if item.detected_at else None
        } for item in items])
    finally:
        session_db.close()


@app.route('/smart_money/detected/delete/<int:ob_id>', methods=['POST'])
def smart_money_detected_delete(ob_id):
    """Видалити Order Block"""
    from models import DetectedOrderBlock
    session_db = db_manager.get_session()
    try:
        item = session_db.query(DetectedOrderBlock).filter_by(id=ob_id).first()
        if item:
            session_db.delete(item)
            session_db.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session_db.close()


# ===== OB SCANNER STATUS & SCHEDULER =====
ob_scanner_state = {
    'is_scanning': False,
    'last_scan': None,
    'next_scan': None,
    'last_found': 0,
    'last_scanned': 0
}

# Глобальний стан для синхронізації між скринерами
screener_lock = {
    'rsi_mfi_running': False,
    'ob_scanner_running': False
}


def get_next_candle_close(tf_minutes: int) -> datetime:
    """Розрахунок часу закриття наступної свічки (синхронізація з біржею)"""
    now = datetime.utcnow()
    
    # Поточна хвилина від початку дня
    current_minutes = now.hour * 60 + now.minute
    
    # Наступне закриття свічки
    next_close_minute = ((current_minutes // tf_minutes) + 1) * tf_minutes
    
    # Якщо перевищує добу, переходимо на наступний день
    if next_close_minute >= 1440:
        next_close_minute = tf_minutes
        next_day = now.date() + timedelta(days=1)
        next_close = datetime(next_day.year, next_day.month, next_day.day,
                             next_close_minute // 60, next_close_minute % 60, 5)
    else:
        next_close = now.replace(
            hour=next_close_minute // 60,
            minute=next_close_minute % 60,
            second=5,  # +5 сек буфер для біржі
            microsecond=0
        )
    
    return next_close


def verify_rsi_mfi_filters(symbol: str, direction: str) -> bool:
    """
    Перевірка монети через RSI/MFI фільтри перед відкриттям угоди.
    Повертає True якщо ринок йде в потрібному напрямку.
    """
    try:
        from rsi_screener import RSIMFIScreener
        
        api_key, api_secret = get_api_credentials()
        from pybit.unified_trading import HTTP
        bybit_session = HTTP(
            testnet=os.environ.get("TESTNET", "false").lower() == "true",
            api_key=api_key,
            api_secret=api_secret
        )
        
        scan_settings = settings.get_all()
        screener = RSIMFIScreener(session=bybit_session, settings=scan_settings)
        
        # Перевіряємо тільки одну монету
        result = screener.check_symbol_filters(symbol, direction)
        
        if result:
            logger.info(f"✅ RSI/MFI filters PASSED for {symbol} ({direction})")
            return True
        else:
            logger.info(f"❌ RSI/MFI filters FAILED for {symbol} ({direction}) - skip trade")
            return False
            
    except Exception as e:
        logger.error(f"RSI/MFI filter check error for {symbol}: {e}")
        # При помилці - не блокуємо угоду
        return True


def execute_ob_trade(ob, bybit_session=None) -> dict:
    """
    Відкриття угоди на основі Order Block
    
    Args:
        ob: DetectedOrderBlock об'єкт або dict з параметрами OB
        bybit_session: Bybit HTTP session (опціонально)
        
    Returns:
        dict з результатом операції
    """
    try:
        from bot import TradingBot
        
        # Отримуємо параметри OB
        if hasattr(ob, 'symbol'):
            # Це DetectedOrderBlock об'єкт
            symbol = ob.symbol
            direction = ob.direction
            entry_price = ob.entry_price
            sl_price = ob.sl_price
            current_price = ob.current_price
            ob_top = ob.ob_top
            ob_bottom = ob.ob_bottom
        else:
            # Це dict
            symbol = ob['symbol']
            direction = ob['direction']
            entry_price = ob['entry_price']
            sl_price = ob['sl_price']
            current_price = ob['current_price']
            ob_top = ob['ob_top']
            ob_bottom = ob['ob_bottom']
        
        # Розрахунок Stop Loss у відсотках
        if direction == 'BUY':
            # Для LONG: SL нижче entry
            sl_percent = abs((entry_price - sl_price) / entry_price * 100)
            action = "Buy"
        else:
            # Для SHORT: SL вище entry
            sl_percent = abs((sl_price - entry_price) / entry_price * 100)
            action = "Sell"
        
        # Отримуємо налаштування ризику
        risk_percent = settings.get('risk_percent', 1.0)
        leverage = settings.get('leverage', 10)
        
        logger.info(f"🚀 Executing OB Trade: {symbol} {direction}")
        logger.info(f"   Entry: {entry_price:.4f}, SL: {sl_price:.4f} ({sl_percent:.2f}%)")
        logger.info(f"   OB Zone: [{ob_bottom:.4f} - {ob_top:.4f}]")
        logger.info(f"   Risk: {risk_percent}%, Leverage: {leverage}x")
        
        # Створюємо бота
        api_key, api_secret = get_api_credentials()
        bot = TradingBot(api_key, api_secret)
        
        # Формуємо дані для place_order
        trade_data = {
            'action': action,
            'symbol': symbol,
            'riskPercent': risk_percent,
            'leverage': leverage,
            'stopLossPercent': sl_percent,
            'entryPrice': current_price  # Використовуємо поточну ціну як entry
        }
        
        # Виконуємо ордер
        result = bot.place_order(trade_data)
        
        if result.get('status') == 'ok':
            logger.info(f"✅ Trade executed: {symbol} {direction} qty={result.get('qty')}")
            return {'status': 'ok', 'trade_result': result}
        else:
            logger.warning(f"⚠️ Trade issue: {symbol} - {result.get('reason', 'Unknown')}")
            return {'status': 'warning', 'reason': result.get('reason'), 'trade_result': result}
            
    except Exception as e:
        logger.error(f"❌ Execute OB trade error for {ob.symbol if hasattr(ob, 'symbol') else ob.get('symbol')}: {e}", exc_info=True)
        return {'status': 'error', 'reason': str(e)}


def scheduled_ob_scan():
    """Scheduled Order Block scan - часте сканування"""
    global ob_scanner_state, screener_lock
    
    # Перевіряємо чи RSI/MFI Screener не працює
    if screener_lock.get('rsi_mfi_running', False):
        logger.info("⏸️ OB Scanner: Paused - RSI/MFI Screener is running")
        # Плануємо наступну спробу через 30 сек
        ob_scanner_state['next_scan'] = (datetime.utcnow() + timedelta(seconds=30)).isoformat()
        return
    
    if ob_scanner_state['is_scanning']:
        logger.warning("OB Scanner: Already scanning, skip")
        return
    
    if not settings.get('ob_auto_scan', False):
        return
    
    logger.info("🔄 OB Scanner: Starting scheduled scan...")
    ob_scanner_state['is_scanning'] = True
    screener_lock['ob_scanner_running'] = True
    
    try:
        # Виконуємо сканування
        from models import SmartMoneyTicker, DetectedOrderBlock
        from order_block_scanner import OrderBlockScanner
        
        session_db = db_manager.get_session()
        try:
            watchlist_items = session_db.query(SmartMoneyTicker).all()
            if not watchlist_items:
                logger.info("OB Scanner: Watchlist is empty")
                return
            
            watchlist = [{'symbol': item.symbol, 'direction': item.direction or 'BUY'} for item in watchlist_items]
            
            # Підключення до Bybit
            api_key, api_secret = get_api_credentials()
            from pybit.unified_trading import HTTP
            bybit_session = HTTP(
                testnet=os.environ.get("TESTNET", "false").lower() == "true",
                api_key=api_key,
                api_secret=api_secret
            )
            
            scan_settings = settings.get_all()
            scanner = OrderBlockScanner(session=bybit_session, settings=scan_settings)
            
            # ===== ЧАСТИНА 1: Моніторинг існуючих OB (ретест) =====
            # Перевіряємо чи ціна торкнулась існуючих OB зон
            existing_obs = session_db.query(DetectedOrderBlock).filter(
                DetectedOrderBlock.status.in_(['Valid', 'Waiting Retest'])
            ).all()
            
            triggered_count = 0
            for ob in existing_obs:
                try:
                    # Отримуємо поточну ціну
                    ticker = bybit_session.get_tickers(category="linear", symbol=ob.symbol)
                    if ticker.get('retCode') != 0:
                        continue
                    
                    ticker_data = ticker.get('result', {}).get('list', [{}])[0]
                    current_price = float(ticker_data.get('lastPrice', 0))
                    high_24h = float(ticker_data.get('highPrice24h', current_price))
                    low_24h = float(ticker_data.get('lowPrice24h', current_price))
                    
                    if current_price <= 0:
                        continue
                    
                    # Перевіряємо чи ціна в зоні OB
                    price_in_zone = (current_price >= ob.ob_bottom and current_price <= ob.ob_top)
                    
                    # Оновлюємо поточну ціну
                    ob.current_price = current_price
                    
                    if price_in_zone and ob.status in ['Valid', 'Waiting Retest']:
                        logger.info(f"🎯 RETEST DETECTED: {ob.symbol} price ${current_price:.4f} in OB zone [{ob.ob_bottom:.4f} - {ob.ob_top:.4f}]")
                        
                        # Ретест НЕ проходить через фільтри! Просто активуємо OB
                        ob.status = 'Triggered'
                        triggered_count += 1
                        
                        # Якщо Execute Trades увімкнено - відкриваємо угоду
                        if settings.get('ob_execute_trades', False):
                            logger.info(f"🚀 Opening trade for {ob.symbol} ({ob.direction}) - RETEST triggered")
                            trade_result = execute_ob_trade(ob, bybit_session)
                            if trade_result.get('status') == 'ok':
                                ob.executed_at = datetime.utcnow()
                                ob.status = 'Executed'
                                ob.trade_result = f"Executed via retest"
                            else:
                                ob.trade_result = f"Failed: {trade_result.get('reason', 'Unknown')}"
                    
                    # Перевірка інвалідації
                    invalidation_method = scan_settings.get('ob_invalidation_method', 'Wick')
                    is_invalidated = False
                    
                    if ob.direction == 'BUY':
                        # Bullish OB інвалідується якщо ціна пішла нижче bottom
                        if invalidation_method == 'Wick':
                            is_invalidated = current_price < ob.ob_bottom
                        else:  # Close
                            is_invalidated = current_price < ob.ob_bottom
                    else:  # SELL
                        # Bearish OB інвалідується якщо ціна пішла вище top
                        if invalidation_method == 'Wick':
                            is_invalidated = current_price > ob.ob_top
                        else:  # Close
                            is_invalidated = current_price > ob.ob_top
                    
                    if is_invalidated and ob.status not in ['Executed', 'Invalidated']:
                        logger.info(f"❌ OB INVALIDATED: {ob.symbol} - price ${current_price:.4f} broke zone")
                        ob.status = 'Invalidated'
                        
                except Exception as e:
                    logger.warning(f"Error checking OB {ob.symbol}: {e}")
                    continue
            
            if triggered_count > 0:
                logger.info(f"🎯 Triggered {triggered_count} OB retests")
            
            # ===== ЧАСТИНА 2: Сканування нових OB =====
            results = scanner.scan_watchlist(watchlist, delay=0.3)
            
            found_count = 0
            for result in results:
                symbol = result['symbol']
                direction = result['direction']
                
                # Перевіряємо чи вже є такий OB (будь-який статус крім Invalidated)
                existing = session_db.query(DetectedOrderBlock).filter(
                    DetectedOrderBlock.symbol == symbol,
                    DetectedOrderBlock.status.in_(['Valid', 'Waiting Retest', 'Triggered'])
                ).first()
                
                if existing:
                    continue
                
                # 🔑 Перевірка RSI/MFI фільтрів ТІЛЬКИ для НОВИХ OB
                # Ретест не проходить через фільтри!
                if settings.get('ob_execute_trades', False):
                    if not verify_rsi_mfi_filters(symbol, direction):
                        logger.info(f"⏭️ Skipping NEW OB {symbol} - RSI/MFI filters not passed")
                        continue
                
                new_ob = DetectedOrderBlock(
                    symbol=symbol,
                    direction=direction,
                    ob_type=result['ob_type'],
                    ob_top=result['ob_top'],
                    ob_bottom=result['ob_bottom'],
                    entry_price=result['entry_price'],
                    sl_price=result['sl_price'],
                    current_price=result['current_price'],
                    atr=result['atr'],
                    status=result['status'],
                    timeframe=scan_settings.get('ob_source_tf', '15')
                )
                session_db.add(new_ob)
                found_count += 1
                logger.info(f"✨ NEW OB: {symbol} {direction} zone [{result['ob_bottom']:.4f} - {result['ob_top']:.4f}]")
                
                # 🚀 Якщо Execute Trades ON та OB готовий до входу (Immediate mode)
                if settings.get('ob_execute_trades', False):
                    entry_mode = scan_settings.get('ob_entry_mode', 'Immediate')
                    
                    if entry_mode == 'Immediate' or result['status'] == 'Triggered':
                        # Для Immediate mode - відкриваємо угоду одразу
                        logger.info(f"🚀 Opening trade for NEW OB: {symbol} ({direction})")
                        trade_result = execute_ob_trade(result, bybit_session)
                        if trade_result.get('status') == 'ok':
                            new_ob.executed_at = datetime.utcnow()
                            new_ob.status = 'Executed'
                            new_ob.trade_result = f"Executed immediately"
                        else:
                            new_ob.trade_result = f"Failed: {trade_result.get('reason', 'Unknown')}"
                    else:
                        # Retest mode - чекаємо ретест
                        logger.info(f"⏳ Waiting for retest: {symbol} ({direction})")
            
            session_db.commit()
            
            # Оновлюємо стан
            ob_scanner_state['last_scan'] = {
                'time': datetime.utcnow().isoformat(),
                'found': found_count,
                'triggered': triggered_count,
                'scanned': len(watchlist)
            }
            ob_scanner_state['last_found'] = found_count
            ob_scanner_state['last_scanned'] = len(watchlist)
            
            logger.info(f"✅ OB Scanner: Found {found_count} new OBs, {triggered_count} retests from {len(watchlist)} symbols")
            
        finally:
            session_db.close()
            
    except Exception as e:
        logger.error(f"❌ OB Scanner error: {e}", exc_info=True)
    finally:
        ob_scanner_state['is_scanning'] = False
        screener_lock['ob_scanner_running'] = False
        
        # Наступне сканування через 1-2 хвилини
        scan_interval = int(settings.get('ob_scan_interval', 60))  # секунди
        ob_scanner_state['next_scan'] = (datetime.utcnow() + timedelta(seconds=scan_interval)).isoformat()


def update_ob_scheduler():
    """Оновлення scheduler при зміні налаштувань"""
    global ob_scheduler_job
    
    try:
        # Видаляємо попередній job якщо є
        if 'ob_scheduler_job' in globals() and ob_scheduler_job:
            ob_scheduler_job.remove()
    except:
        pass
    
    if not settings.get('ob_auto_scan', False):
        logger.info("OB Scheduler: Auto scan disabled")
        ob_scanner_state['next_scan'] = None
        return
    
    # Розрахунок інтервалу
    tf_minutes = int(settings.get('ob_source_tf', '15'))
    
    # Розрахунок наступного запуску
    next_run = get_next_candle_close(tf_minutes)
    ob_scanner_state['next_scan'] = next_run.isoformat()
    
    logger.info(f"OB Scheduler: Next scan at {next_run} (TF: {tf_minutes}m)")


@app.route('/smart_money/status')
def smart_money_status():
    """Статус сканера для UI"""
    from models import SmartMoneyTicker
    session_db = db_manager.get_session()
    try:
        watchlist_count = session_db.query(SmartMoneyTicker).count()
    finally:
        session_db.close()
    
    # Статус scheduler
    scheduler_status = {
        'running': ob_scheduler is not None and ob_scheduler.running if ob_scheduler else False,
        'job_exists': ob_scheduler_job is not None
    }
    
    return jsonify({
        'auto_scan_enabled': settings.get('ob_auto_scan', False),
        'auto_add_enabled': settings.get('ob_auto_add_from_screener', False),
        'is_scanning': ob_scanner_state.get('is_scanning', False),
        'is_paused': screener_lock.get('rsi_mfi_running', False),  # Чи на паузі через RSI Screener
        'next_scan': ob_scanner_state.get('next_scan'),
        'last_scan': ob_scanner_state.get('last_scan'),
        'source_tf': settings.get('ob_source_tf', '15'),
        'scan_interval': settings.get('ob_scan_interval', 60),
        'watchlist_count': watchlist_count,
        'watchlist_limit': settings.get('ob_watchlist_limit', 50),
        'scheduler_status': scheduler_status
    })


@app.route('/smart_money/scan', methods=['POST'])
def smart_money_scan():
    """Сканування watchlist на Order Blocks"""
    from models import SmartMoneyTicker, DetectedOrderBlock
    from order_block_scanner import OrderBlockScanner
    
    session_db = db_manager.get_session()
    try:
        # Отримуємо watchlist
        watchlist_items = session_db.query(SmartMoneyTicker).all()
        if not watchlist_items:
            return jsonify({'status': 'ok', 'found': 0, 'message': 'Watchlist is empty'})
        
        # Підготовка даних
        watchlist = [{'symbol': item.symbol, 'direction': item.direction or 'BUY'} for item in watchlist_items]
        
        # Підключення до Bybit
        api_key, api_secret = get_api_credentials()
        from pybit.unified_trading import HTTP
        bybit_session = HTTP(
            testnet=os.environ.get("TESTNET", "false").lower() == "true",
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Створюємо сканер
        scan_settings = settings.get_all()
        scanner = OrderBlockScanner(session=bybit_session, settings=scan_settings)
        
        # Сканування
        results = scanner.scan_watchlist(watchlist, delay=0.5)
        
        # Зберігаємо знайдені OB
        found_count = 0
        executed_symbols = []
        
        for result in results:
            symbol = result['symbol']
            
            # Перевіряємо чи вже є такий OB
            existing = session_db.query(DetectedOrderBlock).filter_by(
                symbol=symbol,
                status='Valid'
            ).first()
            
            if existing:
                continue
            
            # Зберігаємо новий OB
            new_ob = DetectedOrderBlock(
                symbol=symbol,
                direction=result['direction'],
                ob_type=result['ob_type'],
                ob_top=result['ob_top'],
                ob_bottom=result['ob_bottom'],
                entry_price=result['entry_price'],
                sl_price=result['sl_price'],
                current_price=result['current_price'],
                atr=result['atr'],
                status=result['status'],
                timeframe=scan_settings.get('ob_source_tf', '15')
            )
            session_db.add(new_ob)
            found_count += 1
            
            # Якщо Execute Trades увімкнено і статус Valid
            if settings.get('ob_execute_trades', False) and result['status'] == 'Valid':
                # TODO: Виконати угоду через bot_instance
                # Поки що тільки логуємо
                logger.info(f"Would execute trade: {symbol} {result['direction']}")
                executed_symbols.append(symbol)
        
        session_db.commit()
        
        # Видаляємо виконані символи з watchlist
        for symbol in executed_symbols:
            item = session_db.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
            if item:
                session_db.delete(item)
        session_db.commit()
        
        logger.info(f"OB Scan complete: {found_count} found, {len(executed_symbols)} executed")
        
        return jsonify({
            'status': 'ok',
            'found': found_count,
            'executed': len(executed_symbols),
            'scanned': len(watchlist)
        })
        
    except Exception as e:
        logger.error(f"OB Scan error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        session_db.close()

@app.route('/settings', methods=['GET', 'POST'])
@csrf.exempt  # ⚠️ Тимчасово - потребує CSRF токена в шаблоні пізніше
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

# =====================================================
# ОНОВЛЕНИЙ МАРШРУТ СТОРІНКИ "СТРАТЕГІЯ" (INTEGRATION)
# =====================================================
@app.route('/ob_trend/settings', methods=['GET', 'POST'])
@csrf.exempt  # ⚠️ Тимчасово
def ob_trend_settings_page():
    """
    Тепер ця сторінка відображає Whale Strategy (Автономний модуль).
    """
    if request.method == 'POST':
        # Якщо ви захочете додати збереження налаштувань у майбутньому
        pass
        
    # Отримуємо дані з ядра WhaleCore
    history = whale_core.get_history(limit=50)
    
    return render_template(
        'strategy_ob_trend.html',  # Використовуємо існуючий файл шаблону
        history=history,
        is_scanning=whale_core.is_scanning,
        progress=whale_core.progress,
        status=whale_core.status,
        last_time=whale_core.last_scan_time,
        conf=settings._cache
    )

@app.route('/analyzer/scan', methods=['POST'])
@csrf.exempt  # ⚠️ Тимчасово
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

@app.route('/report')
def report_route(): 
    """
    Маршрут для звіту (Відновлений)
    """
    # Якщо у вас є файл report.py, імпортуємо його
    try:
        from report import render_report_page
        return render_report_page(bot_instance, request)
    except ImportError:
        # Fallback якщо report.py немає
        return "Module report.py not found", 404

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

# ===== RSI/MFI SCREENER ROUTES =====

@app.route('/rsi_screener')
def rsi_screener_page():
    """RSI/MFI Screener сторінка"""
    return render_template('rsi_screener.html')

@app.route('/rsi_screener/settings', methods=['GET'])
def rsi_screener_get_settings():
    """Отримати налаштування RSI Screener"""
    try:
        all_settings = settings.get_all()
        screener_settings = {k: v for k, v in all_settings.items() if k.startswith('screener_')}
        return jsonify(screener_settings)
    except Exception as e:
        logger.error(f"RSI Screener settings error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rsi_screener/settings', methods=['POST'])
def rsi_screener_save_settings():
    """Зберегти налаштування RSI Screener"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Фільтруємо тільки screener налаштування
        screener_data = {k: v for k, v in data.items() if k.startswith('screener_')}
        settings.save_settings(screener_data)
        
        logger.info("RSI Screener settings saved")
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"RSI Screener save error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rsi_screener/scan', methods=['POST'])
def rsi_screener_scan():
    """Запустити скан RSI/MFI"""
    global screener_lock
    
    try:
        from rsi_screener import RSIMFIScreener
        from pybit.unified_trading import HTTP
        
        # Встановлюємо lock - OB Scanner має почекати
        screener_lock['rsi_mfi_running'] = True
        logger.info("🔒 RSI/MFI Screener started - OB Scanner paused")
        
        # Отримуємо налаштування з запиту
        data = request.get_json() or {}
        
        # Об'єднуємо з збереженими налаштуваннями
        all_settings = settings.get_all()
        scan_settings = {**all_settings, **data}
        
        # Підключаємося до Bybit
        api_key, api_secret = get_api_credentials()
        session = HTTP(
            testnet=os.environ.get("TESTNET", "false").lower() == "true",
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Створюємо скринер
        screener = RSIMFIScreener(session=session, settings=scan_settings)
        
        # Запускаємо скан
        results = screener.scan()
        
        logger.info(f"RSI Screener scan complete: {len(results)} matches")
        
        return jsonify({
            "status": "ok",
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"RSI Screener scan error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Знімаємо lock
        screener_lock['rsi_mfi_running'] = False
        logger.info("🔓 RSI/MFI Screener finished - OB Scanner resumed")

# ===== SCHEDULER FOR AUTO SCAN =====

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

ob_scheduler = None
ob_scheduler_job = None

# RSI/MFI Screener scheduler
rsi_scheduler = None
rsi_scheduler_job = None

rsi_screener_state = {
    'is_scanning': False,
    'last_scan': None,
    'next_scan': None,
    'last_found': 0,
    'total_scanned': 0
}

def init_ob_scheduler():
    """Ініціалізація scheduler для Order Block сканування"""
    global ob_scheduler, ob_scheduler_job
    
    if ob_scheduler is not None:
        logger.info("OB Scheduler already initialized")
        return
    
    try:
        ob_scheduler = BackgroundScheduler(daemon=True)
        ob_scheduler.start()
        
        # Реєструємо завершення при виході
        atexit.register(lambda: ob_scheduler.shutdown(wait=False))
        
        # Запускаємо планування якщо Auto Scan увімкнено
        update_ob_scheduler()
        
        logger.info("✅ OB Scheduler initialized and running")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OB Scheduler: {e}")


def update_ob_scheduler():
    """Оновлення scheduler при зміні налаштувань"""
    global ob_scheduler_job
    
    if ob_scheduler is None:
        logger.warning("OB Scheduler not initialized, cannot update")
        return
    
    try:
        # Видаляємо попередній job якщо є
        if ob_scheduler_job is not None:
            try:
                ob_scheduler.remove_job('ob_scan_job')
            except:
                pass
            ob_scheduler_job = None
    except:
        pass
    
    if not settings.get('ob_auto_scan', False):
        logger.info("OB Scheduler: Auto scan disabled")
        ob_scanner_state['next_scan'] = None
        return
    
    # Інтервал сканування (за замовчуванням 60 секунд)
    scan_interval = int(settings.get('ob_scan_interval', 60))
    
    # Створюємо interval trigger для частого сканування
    from apscheduler.triggers.interval import IntervalTrigger
    trigger = IntervalTrigger(seconds=scan_interval)
    
    ob_scheduler_job = ob_scheduler.add_job(
        scheduled_ob_scan,
        trigger=trigger,
        id='ob_scan_job',
        replace_existing=True
    )
    
    # Розрахунок наступного запуску
    next_run = datetime.utcnow() + timedelta(seconds=scan_interval)
    ob_scanner_state['next_scan'] = next_run.isoformat()
    
    logger.info(f"OB Scheduler: Configured for interval={scan_interval}s, next: {next_run}")


# ===== RSI/MFI SCREENER SCHEDULER =====

def scheduled_rsi_mfi_scan():
    """Автоматичне сканування RSI/MFI Screener"""
    global screener_lock, rsi_screener_state
    
    # Перевіряємо чи не працює вже
    if screener_lock.get('rsi_mfi_running', False):
        logger.info("⏸️ RSI/MFI Screener: Already running, skipping")
        return
    
    try:
        from rsi_screener import RSIMFIScreener
        from pybit.unified_trading import HTTP
        
        rsi_screener_state['is_scanning'] = True
        screener_lock['rsi_mfi_running'] = True
        logger.info("🔄 RSI/MFI Auto-Scan started")
        
        # Підключення до Bybit
        api_key, api_secret = get_api_credentials()
        session = HTTP(
            testnet=os.environ.get("TESTNET", "false").lower() == "true",
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Завантажуємо налаштування
        scan_settings = settings.get_all()
        
        # Створюємо скринер
        screener = RSIMFIScreener(session=session, settings=scan_settings)
        
        # Запускаємо скан
        results = screener.scan()
        
        # Оновлюємо стан
        rsi_screener_state['last_scan'] = datetime.utcnow().isoformat()
        rsi_screener_state['last_found'] = len(results)
        rsi_screener_state['total_scanned'] = getattr(screener, '_last_total_scanned', 0)
        
        # Розрахунок наступного запуску
        scan_interval = int(settings.get('screener_scan_interval', 15))
        next_run = datetime.utcnow() + timedelta(minutes=scan_interval)
        rsi_screener_state['next_scan'] = next_run.isoformat()
        
        logger.info(f"✅ RSI/MFI Auto-Scan complete: {len(results)} matches found")
        
    except Exception as e:
        logger.error(f"❌ RSI/MFI Auto-Scan error: {e}")
    finally:
        rsi_screener_state['is_scanning'] = False
        screener_lock['rsi_mfi_running'] = False


def init_rsi_scheduler():
    """Ініціалізація scheduler для RSI/MFI сканування"""
    global rsi_scheduler, rsi_scheduler_job
    
    if rsi_scheduler is not None:
        logger.info("RSI Scheduler already initialized")
        return
    
    try:
        rsi_scheduler = BackgroundScheduler(daemon=True)
        rsi_scheduler.start()
        
        atexit.register(lambda: rsi_scheduler.shutdown(wait=False))
        
        update_rsi_scheduler()
        
        logger.info("✅ RSI/MFI Scheduler initialized and running")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RSI Scheduler: {e}")


def update_rsi_scheduler():
    """Оновлення scheduler RSI/MFI при зміні налаштувань"""
    global rsi_scheduler_job
    
    if rsi_scheduler is None:
        logger.warning("RSI Scheduler not initialized, cannot update")
        return
    
    try:
        if rsi_scheduler_job is not None:
            try:
                rsi_scheduler.remove_job('rsi_scan_job')
            except:
                pass
            rsi_scheduler_job = None
    except:
        pass
    
    if not settings.get('screener_auto_scan', False):
        logger.info("RSI/MFI Scheduler: Auto scan disabled")
        rsi_screener_state['next_scan'] = None
        return
    
    # Інтервал в хвилинах
    scan_interval = int(settings.get('screener_scan_interval', 15))
    
    from apscheduler.triggers.interval import IntervalTrigger
    trigger = IntervalTrigger(minutes=scan_interval)
    
    rsi_scheduler_job = rsi_scheduler.add_job(
        scheduled_rsi_mfi_scan,
        trigger=trigger,
        id='rsi_scan_job',
        replace_existing=True
    )
    
    next_run = datetime.utcnow() + timedelta(minutes=scan_interval)
    rsi_screener_state['next_scan'] = next_run.isoformat()
    
    logger.info(f"RSI/MFI Scheduler: Configured for interval={scan_interval}m, next: {next_run}")


@app.route('/rsi_screener/status', methods=['GET'])
def rsi_screener_status():
    """Статус RSI/MFI Screener"""
    auto_scan = settings.get('screener_auto_scan', False)
    scan_interval = settings.get('screener_scan_interval', 15)
    
    return jsonify({
        'is_scanning': rsi_screener_state['is_scanning'],
        'auto_scan_enabled': auto_scan,
        'scan_interval': scan_interval,
        'last_scan': rsi_screener_state['last_scan'],
        'next_scan': rsi_screener_state['next_scan'] if auto_scan else None,
        'last_found': rsi_screener_state['last_found'],
        'total_scanned': rsi_screener_state['total_scanned'],
        'ob_scanner_paused': screener_lock.get('ob_scanner_running', False)
    })


@app.route('/rsi_screener/auto_scan', methods=['POST'])
def rsi_screener_toggle_auto():
    """Toggle auto scan для RSI/MFI Screener"""
    data = request.get_json() or {}
    enabled = data.get('enabled', False)
    interval = data.get('interval', 15)
    
    settings.save_settings({
        'screener_auto_scan': enabled,
        'screener_scan_interval': interval
    })
    
    update_rsi_scheduler()
    
    return jsonify({
        'status': 'ok',
        'auto_scan': enabled,
        'interval': interval
    })


# ===== АВТОМАТИЧНА ІНІЦІАЛІЗАЦІЯ SCHEDULER =====
# Запускаємо scheduler при імпорті модуля (працює і з Gunicorn)
def _auto_init_scheduler():
    """Автоматична ініціалізація при старті"""
    import threading
    def delayed_init():
        import time
        time.sleep(2)  # Невелика затримка для повної ініціалізації Flask
        init_ob_scheduler()
        init_rsi_scheduler()
    
    thread = threading.Thread(target=delayed_init, daemon=True)
    thread.start()

_auto_init_scheduler()

# === WHALE PRO ROUTES ===
register_whale_pro(app)

# ===== ЗАПУСК =====

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info("starting_flask", host=host, port=port, debug=debug)
    app.run(host=host, port=port, debug=debug)
