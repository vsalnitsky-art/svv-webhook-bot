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

# Імпорти
from bot import BotBybit
from scanner import init_scanner
from settings_manager import SettingsManager
from statistics_service import StatisticsService
from smart_exit_profiles import configure_balanced

# Ініціалізація
app = Flask(__name__)
bot_instance = None
scanner_instance = None
settings = None
stats_service = None

def initialize_bot():
    """Ініціалізація бота"""
    global bot_instance, scanner_instance, settings, stats_service
    
    try:
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
        logger.error(f"❌ Initialization error: {e}")
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


@app.route('/')
def home():
    """Головна сторінка"""
    try:
        if not bot_instance:
            return "Bot not initialized", 500
        
        balance = bot_instance.get_bal()
        positions = scanner_instance.get_active_symbols()
        
        return jsonify({
            'status': 'ok',
            'balance': balance,
            'positions_count': len(positions),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Home error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook для TradingView сигналів"""
    try:
        data = json.loads(request.get_data(as_text=True))
        
        if not bot_instance:
            return jsonify({'error': 'Bot not initialized'}), 500
        
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
            return jsonify({'error': 'Scanner not initialized'}), 500
        
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
                return jsonify({'error': 'Settings not loaded'}), 500
            return jsonify(settings.get_all())
        
        elif request.method == 'POST':
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
            return jsonify({'error': 'Stats not initialized'}), 500
        
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
        'bot': 'initialized' if bot_instance else 'not initialized',
        'scanner': 'initialized' if scanner_instance else 'not initialized',
        'timestamp': datetime.now().isoformat()
    })


# ════════════════════════════════════════════════════════════════════════════════
# ІНІЦІАЛІЗАЦІЯ BOTА (ВИКОНУЄТЬСЯ ЗАВЖДИ, НЕ ТІЛЬКИ В if __name__)
# ════════════════════════════════════════════════════════════════════════════════

logger.info("=" * 60)
logger.info("🚀 STARTING BOT WITH SMART EXIT")
logger.info("=" * 60)

try:
    # Ініціалізуємо бот
    if not initialize_bot():
        logger.error("❌ Failed to initialize bot")
        # Але не експортуємо помилку, щоб gunicorn міг запуститись
    else:
        logger.info("✅ Bot initialized successfully")
        
        # Запускаємо фонові потоки
        threading.Thread(target=sync_trades_periodic, daemon=True).start()
        logger.info("✅ Sync thread started")
        
        threading.Thread(target=scanner_loop, daemon=True).start()
        logger.info("✅ Scanner thread started")
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# ЗАПУСК (для локального тестування)
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("✅ Starting Flask server on port 10000")
    app.run(host='0.0.0.0', port=10000, debug=False)
