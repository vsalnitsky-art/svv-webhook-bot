#!/usr/bin/env python3
"""
Sleeper + Order Block Trading Bot
=================================
Головна точка входу

Запуск:
    python main_bot.py              # Web + Scheduler
    python main_bot.py --web-only   # Тільки Web UI
    python main_bot.py --scan-only  # Одноразовий скан
"""

import os
import sys
import argparse
import signal
import threading
import time
import requests
from datetime import datetime

# Додати кореневу папку до path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.bot_settings import DEFAULT_SETTINGS, BYBIT_CONFIG, WEB_CONFIG
from storage.db_models import init_db
from storage.db_operations import get_db
from alerts.telegram_notifier import get_notifier

# Import app for gunicorn (gunicorn main_bot:app)
from web.flask_app import app


# ===== KEEP-ALIVE THREAD (Anti-Sleep for Render/Heroku) =====

def keep_alive():
    """🫀 Keep-alive для хостів (Render, Heroku) - запобігає засинанню"""
    time.sleep(10)  # Початкова затримка
    
    base_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not base_url:
        port = os.environ.get('PORT', 10000)
        base_url = f'http://127.0.0.1:{port}'
    
    target = f"{base_url}/health"
    print(f"[KEEP-ALIVE] Started, pinging {target} every 5 min")
    
    while True:
        try:
            response = requests.get(target, timeout=10)
            print(f"[KEEP-ALIVE] Ping OK: {response.status_code}")
        except Exception as e:
            print(f"[KEEP-ALIVE] Ping failed: {e}")
        time.sleep(300)  # 5 хвилин


def start_keep_alive():
    """Запустити keep-alive потік"""
    thread = threading.Thread(target=keep_alive, daemon=True, name="KeepAlive")
    thread.start()
    print("[STARTUP] Keep-alive thread started")


def print_banner():
    """Вивести банер"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🌙 SLEEPER + ORDER BLOCK TRADING BOT v1.0 🌙           ║
║                                                           ║
║   Automated crypto trading with:                          ║
║   • Sleeper Detection (accumulation zones)               ║
║   • Order Block Analysis (institutional levels)          ║
║   • Multi-timeframe confirmation                         ║
║   • Risk Management & Position Sizing                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_environment():
    """Перевірити змінні оточення"""
    print("\n[STARTUP] Checking environment...")
    
    warnings = []
    
    # Bybit API
    if not os.getenv('BYBIT_API_KEY'):
        warnings.append("BYBIT_API_KEY not set - using paper trading only")
    if not os.getenv('BYBIT_API_SECRET'):
        warnings.append("BYBIT_API_SECRET not set - using paper trading only")
    
    # Telegram
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        warnings.append("TELEGRAM_BOT_TOKEN not set - notifications disabled")
    if not os.getenv('TELEGRAM_CHAT_ID'):
        warnings.append("TELEGRAM_CHAT_ID not set - notifications disabled")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   • {w}")
    else:
        print("   ✓ All environment variables set")
    
    return len(warnings) == 0


def initialize_database():
    """Ініціалізувати базу даних"""
    print("\n[STARTUP] Initializing database...")
    
    try:
        init_db()
        db = get_db()
        
        # Встановити дефолтні налаштування якщо потрібно
        current_settings = db.get_all_settings()
        
        if 'paper_balance' not in current_settings:
            db.set_setting('paper_balance', DEFAULT_SETTINGS['paper_balance'])
            print(f"   • Set initial paper balance: ${DEFAULT_SETTINGS['paper_balance']}")
        
        if 'execution_mode' not in current_settings:
            db.set_setting('execution_mode', DEFAULT_SETTINGS['execution_mode'].value)
            print(f"   • Set execution mode: {DEFAULT_SETTINGS['execution_mode'].value}")
        
        if 'paper_trading' not in current_settings:
            db.set_setting('paper_trading', DEFAULT_SETTINGS['paper_trading'])
            print(f"   • Paper trading: {DEFAULT_SETTINGS['paper_trading']}")
        
        db.log_event("INFO", "SYSTEM", "Database initialized")
        print("   ✓ Database ready")
        return True
        
    except Exception as e:
        print(f"   ✗ Database error: {e}")
        return False


def run_single_scan():
    """Виконати одноразовий скан"""
    print("\n[SCAN] Running single scan...")
    
    from detection.sleeper_scanner import get_sleeper_scanner
    from detection.ob_scanner import get_ob_scanner
    from detection.signal_merger import get_signal_merger
    
    try:
        # 1. Sleeper scan
        print("\n📊 Scanning for Sleepers...")
        sleeper_scanner = get_sleeper_scanner()
        sleepers = sleeper_scanner.scan()
        
        print(f"\nFound {len(sleepers)} potential sleepers:")
        for s in sleepers[:10]:  # Top 10
            print(f"   • {s['symbol']}: Score {s['total_score']:.1f}, "
                  f"State: {s['state']}, HP: {s['hp']}")
        
        # 2. OB scan для ready sleepers
        ready = [s for s in sleepers if s['state'] == 'READY']
        if ready:
            print(f"\n📦 Scanning Order Blocks for {len(ready)} ready sleepers...")
            ob_scanner = get_ob_scanner()
            
            for s in ready:
                obs = ob_scanner.scan_symbol(s['symbol'])
                if obs:
                    print(f"   {s['symbol']}: {len(obs)} OBs found")
                    for ob in obs:
                        print(f"      - {ob['ob_type']} @ {ob['ob_mid']:.4f}, "
                              f"Quality: {ob['quality_score']:.1f}")
        
        # 3. Signal check
        print("\n🎯 Checking for signals...")
        merger = get_signal_merger()
        signals = merger.check_for_signals()
        
        if signals:
            print(f"\n⚡ {len(signals)} SIGNALS GENERATED:")
            for sig in signals:
                print(f"   • {sig['symbol']} {sig['direction']}")
                print(f"     Entry: ${sig['entry_price']:.4f}")
                print(f"     Confidence: {sig['confidence']:.1f}%")
        else:
            print("   No signals at this time")
        
        print("\n✓ Scan complete")
        return True
        
    except Exception as e:
        print(f"\n✗ Scan error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_web_only():
    """Запустити тільки Web UI"""
    print("\n[WEB] Starting web server only...")
    
    from web.flask_app import app
    
    host = WEB_CONFIG['host']
    port = int(os.getenv('PORT', WEB_CONFIG['port']))
    debug = WEB_CONFIG['debug']
    
    print(f"   • Host: {host}")
    print(f"   • Port: {port}")
    print(f"   • Debug: {debug}")
    print(f"\n🌐 Open http://{host}:{port} in browser\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


def run_full():
    """Запустити Web + Scheduler"""
    print("\n[FULL] Starting web server + background scheduler...")
    
    from web.flask_app import app
    from scheduler.background_jobs import get_scheduler
    
    # Запустити scheduler
    scheduler = get_scheduler()
    scheduler.start()
    
    # Graceful shutdown
    def signal_handler(signum, frame):
        print("\n\n[SHUTDOWN] Stopping...")
        scheduler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Запустити web server
    host = WEB_CONFIG['host']
    port = int(os.getenv('PORT', WEB_CONFIG['port']))
    debug = WEB_CONFIG['debug']
    
    print(f"   • Web: http://{host}:{port}")
    print(f"   • Scheduler: Running")
    print(f"\n🚀 Bot is running! Press Ctrl+C to stop.\n")
    
    # Відправити сповіщення
    notifier = get_notifier()
    notifier.notify_system("Bot started successfully", "INFO")
    
    try:
        # Використовуємо threaded=True для роботи з scheduler
        app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
    finally:
        scheduler.stop()


def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(
        description='Sleeper + Order Block Trading Bot'
    )
    parser.add_argument(
        '--web-only',
        action='store_true',
        help='Run only web interface without scheduler'
    )
    parser.add_argument(
        '--scan-only',
        action='store_true',
        help='Run single scan and exit'
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database and exit'
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    check_environment()
    
    # Initialize database
    if not initialize_database():
        print("\n❌ Failed to initialize database. Exiting.")
        sys.exit(1)
    
    if args.init_db:
        print("\n✓ Database initialized. Exiting.")
        sys.exit(0)
    
    # Run mode
    if args.scan_only:
        success = run_single_scan()
        sys.exit(0 if success else 1)
    elif args.web_only:
        run_web_only()
    else:
        run_full()


# ===== AUTO-START KEEP-ALIVE FOR GUNICORN =====
# Запускається при імпорті модуля (gunicorn main_bot:app)
def _auto_start_keep_alive():
    """Автоматичний запуск keep-alive при імпорті"""
    # Перевіряємо чи це Render/production
    if os.environ.get('RENDER') or os.environ.get('RENDER_EXTERNAL_URL'):
        def delayed_start():
            time.sleep(3)  # Затримка для повної ініціалізації
            start_keep_alive()
        
        thread = threading.Thread(target=delayed_start, daemon=True)
        thread.start()
        print("[STARTUP] Keep-alive scheduled for Render environment")

_auto_start_keep_alive()


if __name__ == '__main__':
    main()
