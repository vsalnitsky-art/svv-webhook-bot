#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Flask app for Render deployment
Does NOT import bot on startup - uses lazy imports only
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

logger.info("=" * 60)
logger.info("🚀 FLASK APP INITIALIZED")
logger.info("=" * 60)


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Trading Bot API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for TradingView signals"""
    try:
        # Lazy import - only when needed
        from bot import BotBybit
        
        data = json.loads(request.get_data(as_text=True))
        
        bot = BotBybit()
        if not hasattr(bot, 'session') or bot.session is None:
            return jsonify({
                'status': 'error',
                'message': 'Bot API not connected',
                'mode': 'demo'
            }), 503
        
        result = bot.place_order(data)
        logger.info(f"✅ Order placed: {result}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/balance', methods=['GET'])
def get_balance():
    """Get account balance"""
    try:
        # Lazy import
        from bot import BotBybit
        
        bot = BotBybit()
        balance = bot.get_bal()
        
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Balance error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    """404 handler"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """500 handler"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


if __name__ == '__main__':
    logger.info("✅ Starting Flask server on port 10000")
    app.run(host='0.0.0.0', port=10000, debug=False)
