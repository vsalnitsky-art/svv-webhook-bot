#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sqlalchemy import desc

# --- ІМПОРТИ МОДУЛІВ ПРОЕКТУ ---
from settings_manager import settings
from models import db_manager, Base, Trade, AnalysisResult, PaperTrade, SmartMoneyTicker
from market_analyzer import market_analyzer

# Lazy import бота, щоб уникнути цикличности, якщо знадобиться, 
# але краще імпортувати клас, а екземпляр створювати за потребою або використовувати singleton
from bot import bot_instance, BybitTradingBot

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ініціалізація Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_123')

# === CONTEXT PROCESSOR ===
# Додає змінні (конфіг, баланс) у всі HTML шаблони автоматично
@app.context_processor
def inject_globals():
    try:
        bal = bot_instance.get_bal()
    except:
        bal = 0.0
    return {
        'conf': settings.get_all(),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'balance': bal
    }

# === ГОЛОВНА СТОРІНКА ===
@app.route('/')
def index():
    days = int(request.args.get('days', 30))
    session = db_manager.get_session()
    try:
        # Фільтр по даті
        cutoff = datetime.now() - timedelta(days=days)
        trades = session.query(Trade).filter(Trade.exit_time >= cutoff).order_by(desc(Trade.exit_time)).all()
        
        longs = sum(1 for t in trades if t.side == 'Long')
        shorts = sum(1 for t in trades if t.side == 'Short')
        pnl = sum(t.pnl for t in trades if t.pnl is not None)
        
        # Отримуємо кількість активних позицій через API Bybit
        active_count = 0
        try:
            positions = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if positions['retCode'] == 0:
                active_count = len([p for p in positions['result']['list'] if float(p['size']) > 0])
        except Exception as e:
            logger.error(f"Bybit connection error: {e}")
        
        return render_template('index.html', 
                             days=days, 
                             trades=trades, 
                             period_pnl=pnl, 
                             longs=longs, 
                             shorts=shorts,
                             active_count=active_count)
    except Exception as e:
        logger.error(f"Index error: {e}")
        return f"Error: {e}", 500
    finally:
        session.close()

# === СКАНЕР АКТИВНИХ ПОЗИЦІЙ ===
@app.route('/scanner')
def scanner_page():
    positions_data = []
    try:
        # Отримуємо реальні дані з біржі
        resp = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if resp['retCode'] == 0:
            for p in resp['result']['list']:
                if float(p['size']) > 0:
                    positions_data.append({
                        'symbol': p['symbol'],
                        'side': 'Buy' if p['side'] == 'Buy' else 'Sell',
                        'size': p['size'],
                        'entry': p['avgPrice'],
                        'pnl': round(float(p['unrealisedPnl']), 2),
                        'rsi': 50, # Заглушка, тут можна підтягнути реальний RSI через strategy
                        'exit_status': 'Monitoring',
                        'exit_details': 'Auto-Strategy Active',
                        'time': datetime.now().strftime('%H:%M')
                    })
    except Exception as e:
        logger.error(f"Scanner error: {e}")

    return render_template('scanner.html', active=positions_data)

# === АНАЛІЗАТОР РИНКУ ===
@app.route('/analyzer')
def analyzer_page():
    # Отримуємо результати з бази даних або пам'яті аналізатора
    results = market_analyzer.get_results()
    return render_template('analyzer.html', 
                         results=results, 
                         is_scanning=market_analyzer.is_scanning,
                         status=market_analyzer.status_message,
                         progress=market_analyzer.progress)

@app.route('/analyzer/scan', methods=['POST'])
def start_scan():
    """Запуск сканування (AJAX)"""
    if market_analyzer.is_scanning:
        return jsonify({'status': 'busy', 'message': 'Scan already in progress'})

    # Оновлюємо налаштування з форми
    form_data = request.form
    
    # Чекбокси
    settings.set('obt_useCloudFilter', form_data.get('obt_useCloudFilter') == 'on')
    settings.set('obt_useObvFilter', form_data.get('obt_useObvFilter') == 'on')
    settings.set('obt_useRsiFilter', form_data.get('obt_useRsiFilter') == 'on')
    settings.set('obt_useOBRetest', form_data.get('obt_useOBRetest') == 'on')
    settings.set('scan_use_min_volume', form_data.get('scan_use_min_volume') == 'on')
    
    # Числа
    if form_data.get('scan_limit'):
        settings.set('scan_limit', int(form_data.get('scan_limit')))
    if form_data.get('scan_min_volume'):
        settings.set('scan_min_volume', float(form_data.get('scan_min_volume')))
        
    settings.save_settings()
    
    # Запуск потоку
    market_analyzer.run_scan_thread()
    return jsonify({'status': 'started'})

@app.route('/analyzer/status')
def scan_status():
    """Статус для прогрес-бару"""
    return jsonify({
        'is_scanning': market_analyzer.is_scanning,
        'progress': market_analyzer.progress,
        'message': market_analyzer.status_message
    })

# === SMART MONEY SIMULATOR ===
@app.route('/smart_money', methods=['GET', 'POST'])
def smart_money_page():
    session = db_manager.get_session()
    try:
        if request.method == 'POST':
            f = request.form
            settings.set('sm_entry_mode', f.get('sm_entry_mode'))
            settings.set('sm_sl_buffer', float(f.get('sm_sl_buffer', 0.2)))
            settings.set('sm_tp_mode', f.get('sm_tp_mode'))
            settings.set('sm_tp_value', float(f.get('sm_tp_value', 3.0)))
            settings.save_settings()
            return redirect(url_for('smart_money_page'))

        watchlist = session.query(SmartMoneyTicker).order_by(desc(SmartMoneyTicker.added_at)).all()
        active_trades = session.query(PaperTrade).filter(PaperTrade.status.in_(['OPEN', 'PENDING'])).order_by(desc(PaperTrade.created_at)).all()
        history_trades = session.query(PaperTrade).filter(PaperTrade.status.in_(['CLOSED_WIN', 'CLOSED_LOSS', 'CANCELED'])).order_by(desc(PaperTrade.closed_at)).limit(50).all()
        
        return render_template('smart_money.html', 
                             watchlist=watchlist, 
                             active_trades=active_trades, 
                             history_trades=history_trades)
    finally:
        session.close()

@app.route('/smart_money/delete/<symbol>', methods=['POST'])
def delete_sm_ticker(symbol):
    session = db_manager.get_session()
    try:
        t = session.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
        if t:
            session.delete(t)
            session.commit()
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error'}), 404
    finally:
        session.close()

# === НАЛАШТУВАННЯ ===
@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    if request.method == 'POST':
        f = request.form
        # Оновлення параметрів
        htf = f.get('htfSelection')
        profile = settings.get('profile', 'BALANCED')
        
        # Це оновлює ВСІ залежні параметри (RSI, SL, TP) через TimeframeParameters
        settings.update_for_timeframe(htf, profile)
        
        # Додаткові ручні налаштування
        settings.set('ltfSelection', f.get('ltfSelection'))
        settings.set('riskPercent', float(f.get('riskPercent')))
        settings.set('leverage', int(f.get('leverage')))
        settings.set('exit_enableStrategy', f.get('exit_enableStrategy') == 'on')
        settings.set('telegram_enabled', f.get('telegram_enabled') == 'on')
        settings.set('telegram_bot_token', f.get('telegram_bot_token'))
        settings.set('telegram_chat_id', f.get('telegram_chat_id'))
        
        settings.save_settings()
        return redirect(url_for('settings_page'))
        
    return render_template('settings.html')

@app.route('/settings/export')
def export_settings():
    return jsonify(settings.get_all())

@app.route('/settings/import', methods=['POST'])
def import_settings():
    try:
        file = request.files['file']
        if file:
            data = json.load(file)
            for k, v in data.items():
                settings.set(k, v)
            settings.save_settings()
    except Exception as e:
        logger.error(f"Import error: {e}")
    return redirect(url_for('settings_page'))

# === WEBHOOK (Для TradingView) ===
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        # Використовуємо вже ініціалізований bot_instance або створюємо новий
        # Перевірка: чи підключений бот
        if not bot_instance.session:
             return jsonify({'status': 'error', 'message': 'Bot not connected'}), 503

        result = bot_instance.place_order(data)
        logger.info(f"Webhook Signal: {data} -> {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 400

# === HEALTH CHECK ===
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()}), 200

# === TEST ROUTES (ДЛЯ full_test.py) ===
@app.route('/test-connection', methods=['GET'])
def test_connection():
    """Тест з'єднання з Bybit"""
    try:
        price = bot_instance.get_price("BTCUSDT")
        if price > 0:
            return jsonify({
                'status': 'ok',
                'btc_price': price,
                'message': 'Connected to Bybit'
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Price is 0'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test-trading', methods=['POST'])
def test_trading_calc():
    """Тест розрахунку позиції (без відкриття угоди)"""
    try:
        data = request.json
        account_balance = float(data.get('accountBalance', 1000))
        risk = float(data.get('riskPercent', 1))
        leverage = int(data.get('leverage', 1))
        
        # Емуляція ціни
        price = 100000 
        
        # Простий розрахунок для тесту
        position_size_usdt = account_balance * (risk / 100) * leverage
        
        return jsonify({
            'status': 'ok',
            'position_size_usdt': position_size_usdt,
            'take_profit_price': price * 1.03,
            'stop_loss_price': price * 0.99
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# === API: CHART DATA ===
@app.route('/api/chart_data/<symbol>')
def get_chart_data(symbol):
    """API для відмальовки графіка в chart_view.html"""
    try:
        tf = settings.get('ltfSelection', '15')
        df = market_analyzer.fetch_candles(symbol, tf, limit=200)
        
        if df is None or df.empty:
            return jsonify({'error': 'No data'}), 404
            
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                'time': int(row['time'].timestamp()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
        
        return jsonify({
            'symbol': symbol,
            'candles': candles,
            # Тут можна додати дані індикаторів, якщо вони будуть розраховані
            'hma_fast': [],
            'hma_slow': []
        })
    except Exception as e:
        logger.error(f"Chart API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chart/<symbol>')
def view_chart(symbol):
    return render_template('chart_view.html', symbol=symbol)

# === ЗАПУСК ===
if __name__ == '__main__':
    # Створюємо таблиці в БД якщо їх немає
    try:
        Base.metadata.create_all(db_manager.engine)
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database init error: {e}")

    # Запускаємо сервер
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)