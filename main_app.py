#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Завантажити .env
load_dotenv()

# Логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ініціалізація Flask
app = Flask(__name__)

# Глобальні змінні
bot_instance = None
scanner_instance = None
settings = None
stats_service = None


def safe_initialize_bot():
    """Безпечна ініціалізація бота (не падає якщо помилка)"""
    global bot_instance, scanner_instance, settings, stats_service
    
    try:
        # Перевіримо API ключі
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not api_key or not api_secret:
            logger.warning("⚠️ BYBIT_API_KEY or BYBIT_API_SECRET not set")
            logger.warning("⚠️ Bot will run in demo mode without real trading")
            return False
        
        from settings_manager import SettingsManager
        from statistics_service import StatisticsService
        from bot import BotBybit
        from scanner import init_scanner
        from smart_exit_profiles import configure_balanced
        
        # Інітимо SettingsManager
        settings = SettingsManager()
        logger.info("✅ Settings loaded")
        
        # Інітимо статистику
        stats_service = StatisticsService()
        logger.info("✅ Statistics service initialized")
        
        # Інітимо бот
        bot_instance = BotBybit()
        logger.info("✅ Bot initialized")
        
        # Інітимо сканер з бот інстансом
        scanner_instance = init_scanner(bot_instance, settings.get_all())
        logger.info("✅ Scanner initialized with Smart Exit")
        
        # Активуємо Smart Exit профіль
        config = configure_balanced()
        logger.info(f"✅ Smart Exit profile activated: {config['name']}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Bot initialization failed: {e}")
        logger.warning("⚠️ Bot will run without real API connection")
        logger.warning("⚠️ API endpoints will return demo data")
        return False


def sync_trades_periodic():
    """Фоновий потік для синхронізації угод"""
    time.sleep(5)
    while True:
        try:
            if bot_instance:
                bot_instance.sync_trades(days=7)
                logger.info("✅ Periodic trades sync completed")
        except Exception as e:
            logger.error(f"❌ Sync error: {e}")
        time.sleep(1800)  # 30 хвилин


def scanner_loop():
    """Фоновий потік для сканера"""
    time.sleep(5)
    while True:
        try:
            if scanner_instance:
                scanner_instance.loop()
        except Exception as e:
            logger.error(f"❌ Scanner loop error: {e}")
        time.sleep(10)  # 10 сек


# ════════════════════════════════════════════════════════════════════════════════
# FLASK МАРШРУТИ
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    """Головна сторінка"""
    return jsonify({
        'status': 'ok',
        'message': 'Trading Bot API',
        'version': '1.0.0',
        'bot_connected': bot_instance is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook для TradingView сигналів"""
    try:
        if not bot_instance:
            return jsonify({'error': 'Bot not connected'}), 503
        
        data = json.loads(request.get_data(as_text=True))
        result = bot_instance.place_order(data)
        logger.info(f"Webhook processed: {result}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/scanner/status', methods=['GET'])
def scanner_status():
    """Статус сканера"""
    try:
        if not scanner_instance:
            return jsonify({'status': 'not_initialized'}), 503
        
        positions = scanner_instance.get_active_symbols()
        return jsonify({
            'status': 'running',
            'positions': len(positions),
            'symbols': [p['symbol'] for p in positions]
        })
    except Exception as e:
        logger.error(f"❌ Scanner status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/settings', methods=['GET', 'POST'])
def get_settings():
    """Отримати/зберегти налаштування"""
    try:
        if request.method == 'GET':
            if not settings:
                return jsonify({'error': 'Settings not loaded'}), 503
            return jsonify(settings.get_all())
        
        elif request.method == 'POST':
            if not settings:
                return jsonify({'error': 'Settings not loaded'}), 503
            
            data = json.loads(request.get_data(as_text=True))
            settings.save_settings(data)
            logger.info(f"Settings updated: {data}")
            return jsonify({'status': 'ok'})
    
    except Exception as e:
        logger.error(f"❌ Settings error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/trades', methods=['GET'])
def get_trades():
    """Отримати статистику угод"""
    try:
        if not stats_service:
            return jsonify({'error': 'Stats not initialized'}), 503
        
        trades = stats_service.get_trades(limit=100)
        return jsonify({
            'total': len(trades),
            'trades': [t.to_dict() if hasattr(t, 'to_dict') else str(t) for t in trades]
        })
    except Exception as e:
        logger.error(f"❌ Trades error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'bot': 'connected' if bot_instance else 'not_connected',
        'scanner': 'initialized' if scanner_instance else 'not_initialized',
        'timestamp': datetime.now().isoformat()
    }), 200


# ════════════════════════════════════════════════════════════════════════════════
# ІНІЦІАЛІЗАЦІЯ (ВИКОНУЄТЬСЯ ПРИ ЗАПУСКУ)
# ════════════════════════════════════════════════════════════════════════════════

logger.info("=" * 60)
logger.info("🚀 STARTING BOT WITH SMART EXIT")
logger.info("=" * 60)

# Безпечна ініціалізація бота
bot_ok = safe_initialize_bot()

if bot_ok:
    logger.info("✅ Bot initialized successfully")
    
    # Запускаємо фонові потоки
    threading.Thread(target=sync_trades_periodic, daemon=True).start()
    logger.info("✅ Sync thread started")
    
    threading.Thread(target=scanner_loop, daemon=True).start()
    logger.info("✅ Scanner thread started")
else:
    logger.warning("⚠️ Running in DEMO MODE (no real API connection)")

logger.info("✅ Flask app ready")


# ════════════════════════════════════════════════════════════════════════════════
# ЛОКАЛЬНИЙ ЗАПУСК
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("✅ Starting Flask server on port 10000")
    app.run(host='0.0.0.0', port=10000, debug=False)

