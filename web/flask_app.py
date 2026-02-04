"""
Flask Web Application - Sleeper OB Bot Dashboard
Main web server with routes and templates
"""

import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
from functools import wraps

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.bot_settings import DEFAULT_SETTINGS, ExecutionMode
from storage.db_operations import get_db
from storage.db_models import init_db


def create_app():
    """Create and configure Flask application"""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    )
    
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'sleeper-ob-bot-secret-key-change-me')
    
    # Context processor for templates
    @app.context_processor
    def inject_now():
        return {'now': datetime.utcnow()}
    
    # Initialize database
    init_db()
    
    # Register routes
    register_routes(app)
    
    # Register diagnostic blueprint
    from web.diagnostic import diagnostic_bp
    app.register_blueprint(diagnostic_bp)
    
    # Test Binance API connectivity on startup (для сканування)
    try:
        print("[APP] Testing Binance Futures API connectivity...")
        from core.binance_connector import get_binance_connector
        binance = get_binance_connector()
        if binance.test_connection():
            tickers = binance.get_tickers()
            print(f"[APP] ✓ Binance Futures API working: {len(tickers)} tickers available")
        else:
            print("[APP] ⚠ Binance Futures API connection test failed")
    except Exception as e:
        print(f"[APP] ✗ Binance API test failed: {e}")
    
    # Test Bybit API connectivity on startup (для торгівлі)
    try:
        print("[APP] Testing Bybit API connectivity...")
        from core.bybit_connector import get_connector
        connector = get_connector()
        tickers = connector.get_tickers()
        if tickers:
            print(f"[APP] ✓ Bybit API working: {len(tickers)} tickers available")
        else:
            print("[APP] ⚠ Bybit API returned empty tickers list")
    except Exception as e:
        print(f"[APP] ✗ Bybit API test failed: {e}")
    
    # Start background scheduler (if enabled)
    enable_scheduler = os.getenv('ENABLE_SCHEDULER', 'true').lower() in ('true', '1', 'yes')
    if enable_scheduler:
        try:
            from scheduler.background_jobs import get_scheduler
            scheduler = get_scheduler()
            if not scheduler.is_running:
                scheduler.start()
                print("[APP] Background scheduler started automatically")
        except Exception as e:
            print(f"[APP] Failed to start scheduler: {e}")
    
    return app


def register_routes(app):
    """Register all web routes"""
    
    @app.route('/')
    def index():
        """Main dashboard"""
        db = get_db()
        
        # Get statistics
        trade_stats = db.get_trade_stats(days=30)
        sleepers = db.get_sleepers()
        open_trades = db.get_open_trades()
        recent_events = db.get_recent_events(limit=20)
        
        # Get settings
        settings = db.get_all_settings()
        execution_mode = settings.get('execution_mode', DEFAULT_SETTINGS['execution_mode'])
        paper_trading = settings.get('paper_trading', DEFAULT_SETTINGS['paper_trading'])
        paper_balance = settings.get('paper_balance', DEFAULT_SETTINGS['paper_balance'])
        
        # Count sleepers by state (sleepers are dicts)
        sleeper_counts = {
            'total': len(sleepers),
            'watching': len([s for s in sleepers if s.get('state') == 'WATCHING']),
            'building': len([s for s in sleepers if s.get('state') == 'BUILDING']),
            'ready': len([s for s in sleepers if s.get('state') == 'READY'])
        }
        
        return render_template('dashboard.html',
            trade_stats=trade_stats,
            sleepers=sleepers[:10],  # Top 10
            sleeper_counts=sleeper_counts,
            open_trades=open_trades,
            recent_events=recent_events,
            execution_mode=execution_mode,
            paper_trading=paper_trading,
            paper_balance=paper_balance,
            now=datetime.utcnow()
        )
    
    @app.route('/sleepers')
    def sleepers_page():
        """Sleeper candidates page"""
        db = get_db()
        sleepers = db.get_sleepers()
        
        # Sort by total_score desc (sleepers are dicts)
        sleepers.sort(key=lambda x: x.get('total_score') or 0, reverse=True)
        
        return render_template('sleepers.html',
            sleepers=sleepers,
            now=datetime.utcnow()
        )
    
    @app.route('/orderblocks')
    def orderblocks_page():
        """Order blocks page"""
        db = get_db()
        
        # Get active OBs
        obs = db.get_orderblocks(status='ACTIVE', limit=50)
        
        return render_template('orderblocks.html',
            orderblocks=obs,
            now=datetime.utcnow()
        )
    
    @app.route('/trades')
    def trades_page():
        """Trade history page"""
        db = get_db()
        
        page = request.args.get('page', 1, type=int)
        per_page = 20
        
        trades = db.get_trades(limit=per_page * page)
        total_trades = len(trades)
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        trades_page = trades[start:end]
        
        # Stats
        trade_stats = db.get_trade_stats(days=30)
        
        return render_template('trades.html',
            trades=trades_page,
            trade_stats=trade_stats,
            page=page,
            total_pages=(total_trades + per_page - 1) // per_page,
            now=datetime.utcnow()
        )
    
    @app.route('/settings')
    def settings_page():
        """Bot settings page"""
        db = get_db()
        settings = db.get_all_settings()
        
        # Merge with defaults
        for key, value in DEFAULT_SETTINGS.items():
            if key not in settings:
                settings[key] = value
        
        return render_template('settings.html',
            settings=settings,
            execution_modes=[e.value for e in ExecutionMode]
        )
    
    @app.route('/signals')
    def signals_page():
        """Pending signals page"""
        from detection.signal_merger import get_signal_merger
        
        merger = get_signal_merger()
        pending = merger.get_pending_signals()
        ready = merger.get_ready_for_entry()
        
        return render_template('signals.html',
            pending_signals=pending,
            ready_signals=ready,
            now=datetime.utcnow()
        )


# ============== API ROUTES ==============

def register_api_routes(app):
    """Register API endpoints"""
    
    @app.route('/health')
    @app.route('/api/health')
    def api_health():
        """Health check endpoint for Render"""
        return jsonify({
            'status': 'ok',
            'service': 'sleeper-ob-bot',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/stats')
    def api_stats():
        """Get dashboard statistics"""
        db = get_db()
        
        trade_stats = db.get_trade_stats(days=30)
        sleepers = db.get_sleepers()
        open_trades = db.get_open_trades()
        
        settings = db.get_all_settings()
        paper_balance = settings.get('paper_balance', DEFAULT_SETTINGS['paper_balance'])
        
        return jsonify({
            'success': True,
            'data': {
                'trade_stats': trade_stats,
                'sleeper_count': len(sleepers),
                'ready_sleepers': len([s for s in sleepers if s.get('state') == 'READY']),
                'open_trades': len(open_trades),
                'paper_balance': paper_balance
            }
        })
    
    @app.route('/api/sleepers')
    def api_sleepers():
        """Get all sleeper candidates"""
        db = get_db()
        sleepers = db.get_sleepers()
        
        # Sleepers are already dicts from to_dict()
        return jsonify({'success': True, 'data': sleepers})
    
    @app.route('/api/orderblocks')
    def api_orderblocks():
        """Get active order blocks"""
        db = get_db()
        
        # Get OBs as dicts
        obs = db.get_orderblocks(status='ACTIVE', limit=50)
        
        # Convert datetime to string for JSON
        for ob in obs:
            if ob.get('created_at') and hasattr(ob['created_at'], 'isoformat'):
                ob['created_at'] = ob['created_at'].isoformat()
            if ob.get('expires_at') and hasattr(ob['expires_at'], 'isoformat'):
                ob['expires_at'] = ob['expires_at'].isoformat()
        
        return jsonify({'success': True, 'data': obs})
    
    @app.route('/api/trades')
    def api_trades():
        """Get recent trades"""
        db = get_db()
        
        limit = request.args.get('limit', 20, type=int)
        status = request.args.get('status')
        trades = db.get_trades(status=status, limit=limit)
        
        return jsonify({'success': True, 'data': trades})
    
    @app.route('/api/signals')
    def api_signals():
        """Get pending signals"""
        from detection.signal_merger import get_signal_merger
        
        merger = get_signal_merger()
        signals = merger.get_pending_signals()
        
        return jsonify({'success': True, 'data': signals})
    
    @app.route('/api/settings', methods=['GET', 'POST'])
    def api_settings():
        """Get or update settings"""
        db = get_db()
        
        if request.method == 'GET':
            settings = db.get_all_settings()
            for key, value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = value
            return jsonify({'success': True, 'data': settings})
        
        else:  # POST
            data = request.get_json() or {}
            
            # Accept all settings, not just those in DEFAULT_SETTINGS
            for key, value in data.items():
                db.set_setting(key, value)
            
            # Reload alert settings in notifier if any alert settings were changed
            alert_keys = [k for k in data.keys() if k.startswith('alert_')]
            if alert_keys:
                from alerts.telegram_notifier import get_notifier
                notifier = get_notifier()
                notifier.load_alert_settings(db)
            
            db.log_event(
                message=f'Settings updated: {len(data)} parameters',
                level='INFO',
                category='SYSTEM'
            )
            return jsonify({'success': True, 'message': 'Settings updated'})
    
    @app.route('/api/settings/reset', methods=['POST'])
    def api_settings_reset():
        """Reset all settings to defaults"""
        db = get_db()
        
        # Set all defaults
        for key, value in DEFAULT_SETTINGS.items():
            db.set_setting(key, value)
        
        # Add additional defaults not in DEFAULT_SETTINGS
        additional_defaults = {
            # Sleeper detection
            'sleeper_min_score': 40,
            'sleeper_building_score': 50,
            'sleeper_ready_score': 60,
            'sleeper_min_volume': 75000000,
            'sleeper_timeframe': '240',
            'weight_fuel': 30,
            'weight_volatility': 25,
            'weight_price': 25,
            'weight_liquidity': 20,
            # Order Block (Pine Script params)
            'ob_swing_length': 5,
            'ob_max_atr_mult': 3.5,
            'ob_zone_count': 'Low',
            'ob_end_method': 'Wick',
            'ob_max_count': 30,
            'ob_timeframes': '15,5',
            'ob_min_quality': 60,
            'ob_signal_quality': 70,
            # Trading
            'atr_tp_multiplier': 3,
            'trailing_stop_enabled': True,
            'tp1_pct': 50,
            'tp2_pct': 25,
            'tp3_pct': 25,
            'max_daily_loss_pct': 5,
            'max_daily_trades': 10,
            'max_position_pct': 20,
        }
        
        for key, value in additional_defaults.items():
            db.set_setting(key, value)
        
        # Trend analyzer defaults
        trend_defaults = {
            'use_trend_filter': True,
            'trend_timeframe': '240',
            'min_trend_score': 65,
            'allow_signals_without_trend': False,
            'trading_mode': 'SWING',
        }
        
        for key, value in trend_defaults.items():
            db.set_setting(key, value)
        
        # v8.2.6: Direction Engine defaults
        direction_defaults = {
            'dir_weight_smc': 40,
            'dir_weight_structure': 20,
            'dir_weight_momentum': 20,
            'dir_weight_derivatives': 20,
            'bias_threshold_long': 0.10,
            'bias_threshold_short': -0.10,
        }
        
        for key, value in direction_defaults.items():
            db.set_setting(key, value)
        
        db.log_event(
            message='Settings reset to defaults',
            level='INFO',
            category='SYSTEM'
        )
        return jsonify({'success': True, 'message': 'Settings reset to defaults'})
    
    @app.route('/api/telegram/reload', methods=['POST'])
    def api_telegram_reload():
        """Reload Telegram configuration from environment"""
        from alerts.telegram_notifier import get_notifier
        
        notifier = get_notifier()
        enabled = notifier.reload()
        
        return jsonify({
            'success': True,
            'enabled': enabled,
            'message': 'Telegram enabled' if enabled else 'Telegram disabled (check env vars)'
        })
    
    @app.route('/api/telegram/test', methods=['POST'])
    def api_telegram_test():
        """Send test message to Telegram"""
        from alerts.telegram_notifier import get_notifier
        
        notifier = get_notifier()
        
        # Reload first to pick up any new env vars
        notifier.reload()
        
        if not notifier.enabled:
            return jsonify({
                'success': False,
                'error': 'Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to environment.'
            })
        
        # Send test message
        test_message = """
🧪 <b>TEST MESSAGE</b>

✅ Telegram integration working!
📊 SVV Webhook Bot v4.2
⏱ """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        result = notifier.send_sync(test_message.strip())
        
        return jsonify({
            'success': result,
            'message': 'Test message sent!' if result else 'Failed to send message'
        })
    
    @app.route('/api/scan/sleepers', methods=['POST'])
    def api_scan_sleepers():
        """Trigger sleeper scan (legacy v2)"""
        from detection.sleeper_scanner import get_sleeper_scanner
        
        scanner = get_sleeper_scanner()
        results = scanner.scan()
        
        return jsonify({
            'success': True,
            'message': f'Scanned {len(results)} symbols',
            'data': {
                'total': len(results),
                'ready': len([r for r in results if r.get('state') == 'READY']),
                'building': len([r for r in results if r.get('state') == 'BUILDING'])
            }
        })
    
    @app.route('/api/scan/sleepers/v3', methods=['POST'])
    def api_scan_sleepers_v3():
        """Trigger sleeper scan with 5-day strategy (v3)"""
        from detection.sleeper_scanner_v3 import get_sleeper_scanner_v3
        
        scanner = get_sleeper_scanner_v3()
        results = scanner.run_scan()
        
        # Count by state
        states = {}
        for r in results:
            state = r.get('state', 'UNKNOWN')
            states[state] = states.get(state, 0) + 1
        
        return jsonify({
            'success': True,
            'message': f'5-Day Strategy: Scanned {len(results)} candidates',
            'data': {
                'total': len(results),
                'ready': states.get('READY', 0),
                'building': states.get('BUILDING', 0),
                'watching': states.get('WATCHING', 0),
                'triggered': states.get('TRIGGERED', 0),
                'by_state': states
            }
        })
    
    @app.route('/api/scan/orderblocks', methods=['POST'])
    def api_scan_orderblocks():
        """Trigger OB scan for a symbol"""
        data = request.get_json() or {}
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        from detection.ob_scanner import get_ob_scanner
        
        scanner = get_ob_scanner()
        obs = scanner.scan_symbol(symbol)
        
        return jsonify({
            'success': True,
            'message': f'Found {len(obs)} order blocks for {symbol}',
            'data': obs
        })
    
    @app.route('/api/sleepers/clear-bad', methods=['POST'])
    def api_clear_bad_sleepers():
        """Remove sleepers with poor data quality"""
        db = get_db()
        count = db.remove_low_quality_sleepers()
        
        return jsonify({
            'success': True,
            'message': f'Removed {count} low-quality sleepers',
            'removed': count
        })
    
    @app.route('/api/sleepers/clear-all', methods=['POST'])
    def api_clear_all_sleepers():
        """Remove ALL sleepers for fresh scan"""
        db = get_db()
        count = db.clear_all_sleepers()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {count} sleepers',
            'removed': count
        })
    
    # ==========================================
    # BLACKLIST API (v8.2.2)
    # ==========================================
    
    @app.route('/api/blacklist', methods=['GET'])
    def api_get_blacklist():
        """Отримати список заблокованих монет"""
        db = get_db()
        entries = db.get_blacklist_full()
        return jsonify({
            'success': True,
            'count': len(entries),
            'data': entries
        })
    
    @app.route('/api/blacklist/add', methods=['POST'])
    def api_add_to_blacklist():
        """Додати монету в blacklist"""
        db = get_db()
        data = request.json or {}
        
        symbol = data.get('symbol', '').upper().strip()
        reason = data.get('reason', 'MANUAL')
        note = data.get('note', '')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        success = db.add_to_blacklist(symbol, reason=reason, note=note)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{symbol} added to blacklist'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'{symbol} already in blacklist'
            }), 400
    
    @app.route('/api/blacklist/remove', methods=['POST'])
    def api_remove_from_blacklist():
        """Видалити монету з blacklist"""
        db = get_db()
        data = request.json or {}
        
        symbol = data.get('symbol', '').upper().strip()
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        success = db.remove_from_blacklist(symbol)
        
        return jsonify({
            'success': success,
            'message': f'{symbol} removed from blacklist' if success else f'{symbol} not found'
        })
    
    @app.route('/api/blacklist/clear', methods=['POST'])
    def api_clear_blacklist():
        """Очистити весь blacklist"""
        db = get_db()
        count = db.clear_blacklist()
        return jsonify({
            'success': True,
            'message': f'Cleared {count} entries',
            'removed': count
        })
    
    @app.route('/api/blacklist/init-defaults', methods=['POST'])
    def api_init_default_blacklist():
        """Ініціалізувати стандартний blacklist (stablecoins, wrapped tokens)"""
        db = get_db()
        
        # Default blacklist entries
        defaults = [
            ('USDCUSDT', 'STABLECOIN', 'USD Coin stablecoin'),
            ('FDUSDUSDT', 'STABLECOIN', 'First Digital USD stablecoin'),
            ('TUSDUSDT', 'STABLECOIN', 'TrueUSD stablecoin'),
            ('USDPUSDT', 'STABLECOIN', 'Pax Dollar stablecoin'),
            ('DAIUSDT', 'STABLECOIN', 'DAI stablecoin'),
            ('EURUSDT', 'STABLECOIN', 'Euro stablecoin'),
        ]
        
        added = 0
        for symbol, reason, note in defaults:
            if db.add_to_blacklist(symbol, reason=reason, note=note):
                added += 1
        
        return jsonify({
            'success': True,
            'message': f'Added {added} default entries',
            'added': added
        })
    
    @app.route('/api/sleepers/cleanup-blacklisted', methods=['POST'])
    def api_cleanup_blacklisted_sleepers():
        """Видалити sleepers що є в blacklist"""
        db = get_db()
        removed = db.remove_blacklisted_sleepers()
        return jsonify({
            'success': True,
            'message': f'Removed {removed} blacklisted sleepers',
            'removed': removed
        })
    
    @app.route('/api/sleepers/cleanup-duplicates', methods=['POST'])
    def api_cleanup_duplicate_sleepers():
        """Видалити дублікати sleepers"""
        db = get_db()
        removed = db.remove_duplicate_sleepers()
        return jsonify({
            'success': True,
            'message': f'Removed {removed} duplicate sleepers',
            'removed': removed
        })
    
    # ==========================================
    # END BLACKLIST API
    # ==========================================
    
    @app.route('/api/scan/full', methods=['POST'])
    def api_full_scan():
        """Run full scan cycle: Sleepers → OBs → Signals"""
        from detection.signal_merger import get_signal_merger
        
        merger = get_signal_merger()
        signals = merger.run_full_scan()
        
        return jsonify({
            'success': True,
            'message': f'Full scan complete, {len(signals)} signals generated',
            'data': signals
        })
    
    @app.route('/api/signal/confirm/<signal_id>', methods=['POST'])
    def api_confirm_signal(signal_id):
        """Confirm a pending signal"""
        from detection.signal_merger import get_signal_merger
        
        merger = get_signal_merger()
        result = merger.confirm_signal(signal_id)
        
        if result:
            return jsonify({'success': True, 'message': 'Signal confirmed and executed'})
        return jsonify({'success': False, 'error': 'Signal not found or already processed'}), 404
    
    @app.route('/api/signal/reject/<signal_id>', methods=['POST'])
    def api_reject_signal(signal_id):
        """Reject a pending signal"""
        from detection.signal_merger import get_signal_merger
        
        merger = get_signal_merger()
        result = merger.reject_signal(signal_id)
        
        if result:
            return jsonify({'success': True, 'message': 'Signal rejected'})
        return jsonify({'success': False, 'error': 'Signal not found'}), 404
    
    @app.route('/api/trade/close/<int:trade_id>', methods=['POST'])
    def api_close_trade(trade_id):
        """Close a specific trade"""
        from trading.order_executor import get_executor
        
        executor = get_executor()
        
        # Get trade
        db = get_db()
        from storage.db_models import Trade, get_session
        session = get_session()
        trade = session.query(Trade).filter_by(id=trade_id).first()
        
        if not trade:
            session.close()
            return jsonify({'success': False, 'error': 'Trade not found'}), 404
        
        result = executor.close_position(trade.symbol)
        session.close()
        
        if result:
            return jsonify({'success': True, 'message': 'Trade closed'})
        return jsonify({'success': False, 'error': 'Failed to close trade'}), 500
    
    @app.route('/api/trade/close-all', methods=['POST'])
    def api_close_all_trades():
        """Close all open trades"""
        from trading.order_executor import get_executor
        
        executor = get_executor()
        results = executor.close_all_positions()
        
        return jsonify({
            'success': True,
            'message': f'Closed {len(results)} trades',
            'data': results
        })
    
    @app.route('/api/events')
    def api_events():
        """Get recent events"""
        db = get_db()
        
        limit = request.args.get('limit', 50, type=int)
        category = request.args.get('category')
        
        events = db.get_recent_events(limit=limit, category=category)
        
        # events already are dicts from to_dict()
        return jsonify({'success': True, 'data': events})
    
    # ===== SCHEDULER API =====
    
    @app.route('/api/scheduler/status')
    def api_scheduler_status():
        """Get scheduler status and job stats"""
        from scheduler.background_jobs import get_scheduler
        
        scheduler = get_scheduler()
        
        jobs = []
        for job in scheduler.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'is_running': scheduler.is_running,
                'jobs': jobs,
                'job_stats': scheduler.job_stats
            }
        })
    
    @app.route('/api/scheduler/start', methods=['POST'])
    def api_scheduler_start():
        """Start the scheduler"""
        from scheduler.background_jobs import get_scheduler
        
        scheduler = get_scheduler()
        if scheduler.is_running:
            return jsonify({'success': False, 'message': 'Scheduler already running'})
        
        scheduler.start()
        return jsonify({'success': True, 'message': 'Scheduler started'})
    
    @app.route('/api/scheduler/stop', methods=['POST'])
    def api_scheduler_stop():
        """Stop the scheduler"""
        from scheduler.background_jobs import get_scheduler
        
        scheduler = get_scheduler()
        if not scheduler.is_running:
            return jsonify({'success': False, 'message': 'Scheduler not running'})
        
        scheduler.stop()
        return jsonify({'success': True, 'message': 'Scheduler stopped'})
    
    @app.route('/api/scheduler/trigger/<job_id>', methods=['POST'])
    def api_scheduler_trigger(job_id):
        """Manually trigger a specific job"""
        from scheduler.background_jobs import get_scheduler
        
        scheduler = get_scheduler()
        result = scheduler.trigger_job(job_id)
        
        if result:
            return jsonify({'success': True, 'message': f'Job {job_id} triggered'})
        return jsonify({'success': False, 'error': f'Job {job_id} not found'}), 404
    
    # =========================================
    # TREND ANALYSIS API
    # =========================================
    
    @app.route('/api/trend/<symbol>', methods=['GET'])
    def api_trend_analysis(symbol):
        """
        Get trend analysis for a symbol.
        Returns 4-component TrendScore with regime classification.
        """
        from detection.trend_analyzer import get_trend_analyzer
        
        analyzer = get_trend_analyzer()
        timeframe = request.args.get('timeframe', '240')  # 4H default
        
        result = analyzer.analyze(symbol, timeframe)
        
        if result:
            return jsonify({
                'success': True,
                'data': {
                    'symbol': result.symbol,
                    'timeframe': result.timeframe,
                    'total_score': round(result.total_score, 1),
                    'regime': result.regime.value,
                    'direction': result.overall_direction.value,
                    'components': {
                        'structure': {
                            'score': round(result.structure_score, 1),
                            'direction': result.structure_direction.value,
                            'weight': '30%',
                            'details': result.details.get('structure', {})
                        },
                        'volatility': {
                            'score': round(result.volatility_score, 1),
                            'weight': '25%',
                            'details': result.details.get('volatility', {})
                        },
                        'acceptance': {
                            'score': round(result.acceptance_score, 1),
                            'weight': '25%',
                            'details': result.details.get('acceptance', {})
                        },
                        'momentum': {
                            'score': round(result.momentum_score, 1),
                            'weight': '20%',
                            'details': result.details.get('momentum', {})
                        }
                    },
                    'calculated_at': result.calculated_at.isoformat()
                }
            })
        
        return jsonify({
            'success': False,
            'error': f'Could not analyze trend for {symbol}'
        }), 400
    
    @app.route('/api/trend/batch', methods=['POST'])
    def api_trend_batch():
        """
        Get trend analysis for multiple symbols.
        Request body: {"symbols": ["BTCUSDT", "ETHUSDT", ...]}
        """
        from detection.trend_analyzer import get_trend_analyzer
        
        data = request.get_json() or {}
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '240')
        
        if not symbols:
            return jsonify({'success': False, 'error': 'No symbols provided'}), 400
        
        analyzer = get_trend_analyzer()
        results = {}
        
        for symbol in symbols[:20]:  # Limit to 20
            result = analyzer.analyze(symbol, timeframe)
            if result:
                results[symbol] = {
                    'score': round(result.total_score, 1),
                    'regime': result.regime.value,
                    'direction': result.overall_direction.value
                }
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
    
    @app.route('/api/trend/check', methods=['POST'])
    def api_trend_check():
        """
        Check if trading is allowed for symbol + direction.
        Request body: {"symbol": "BTCUSDT", "direction": "LONG"}
        """
        from detection.trend_analyzer import get_trend_analyzer
        
        data = request.get_json() or {}
        symbol = data.get('symbol')
        direction = data.get('direction')
        
        if not symbol or not direction:
            return jsonify({'success': False, 'error': 'symbol and direction required'}), 400
        
        analyzer = get_trend_analyzer()
        allowed = analyzer.is_tradeable(symbol, direction)
        result = analyzer.analyze(symbol)
        
        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'direction': direction,
                'allowed': allowed,
                'regime': result.regime.value if result else 'UNKNOWN',
                'score': round(result.total_score, 1) if result else 0,
                'reason': 'Trend aligned' if allowed else 'Trend filter blocked'
            }
        })
    
    @app.route('/api/trend/sleepers', methods=['GET'])
    def api_trend_sleepers():
        """
        Get trend analysis for all READY sleepers.
        Useful for dashboard display.
        """
        from detection.trend_analyzer import get_trend_analyzer
        
        db = get_db()
        analyzer = get_trend_analyzer()
        
        ready_sleepers = db.get_sleepers(state='READY')
        results = []
        
        for sleeper in ready_sleepers[:30]:  # Limit
            symbol = sleeper['symbol']
            trend = analyzer.analyze(symbol)
            
            if trend:
                results.append({
                    'symbol': symbol,
                    'sleeper_direction': sleeper['direction'],
                    'sleeper_score': sleeper['total_score'],
                    'trend_score': round(trend.total_score, 1),
                    'trend_regime': trend.regime.value,
                    'trend_direction': trend.overall_direction.value,
                    'allowed': (
                        (trend.regime.value == 'BULLISH' and sleeper['direction'] == 'LONG') or
                        (trend.regime.value == 'BEARISH' and sleeper['direction'] == 'SHORT')
                    )
                })
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'allowed_count': len([r for r in results if r['allowed']])
        })
    
    # =========================================
    # DATABASE CLEANUP API
    # =========================================
    
    @app.route('/api/cleanup/orderblocks', methods=['POST'])
    def api_cleanup_orderblocks():
        """Delete all order blocks from database"""
        from storage.db_models import OrderBlock, get_session
        
        session = get_session()
        try:
            count = session.query(OrderBlock).delete()
            session.commit()
            
            db = get_db()
            db.log_event(f"Deleted {count} order blocks", level='WARN', category='CLEANUP')
            
            return jsonify({
                'success': True,
                'message': f'Deleted {count} order blocks',
                'deleted': count
            })
        except Exception as e:
            session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            session.close()
    
    @app.route('/api/cleanup/sleepers', methods=['POST'])
    def api_cleanup_sleepers():
        """Delete all sleepers from database"""
        from storage.db_models import SleeperCandidate, get_session
        
        session = get_session()
        try:
            count = session.query(SleeperCandidate).delete()
            session.commit()
            
            db = get_db()
            db.log_event(f"Deleted {count} sleepers", level='WARN', category='CLEANUP')
            
            return jsonify({
                'success': True,
                'message': f'Deleted {count} sleepers',
                'deleted': count
            })
        except Exception as e:
            session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            session.close()
    
    @app.route('/api/cleanup/all', methods=['POST'])
    def api_cleanup_all():
        """Delete all data from all tables (except settings and events)"""
        from storage.db_models import OrderBlock, SleeperCandidate, Trade, get_session
        
        session = get_session()
        try:
            ob_count = session.query(OrderBlock).delete()
            sleeper_count = session.query(SleeperCandidate).delete()
            trade_count = session.query(Trade).filter(Trade.status != 'OPEN').delete()
            session.commit()
            
            db = get_db()
            db.log_event(
                f"Full cleanup: {ob_count} OBs, {sleeper_count} sleepers, {trade_count} closed trades",
                level='WARN', category='CLEANUP'
            )
            
            return jsonify({
                'success': True,
                'message': 'Database cleaned',
                'deleted': {
                    'orderblocks': ob_count,
                    'sleepers': sleeper_count,
                    'trades': trade_count
                }
            })
        except Exception as e:
            session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            session.close()
    
    @app.route('/api/debug/sleeper/<symbol>', methods=['GET'])
    def api_debug_sleeper(symbol):
        """Debug endpoint - show raw data for sleeper analysis"""
        from core import get_fetcher
        
        fetcher = get_fetcher()
        
        # Get raw data
        funding = fetcher.get_funding_rate(symbol)
        oi_change = fetcher.get_oi_change(symbol, hours=4)
        klines_4h = fetcher.get_klines(symbol, '240', limit=50)
        
        # Calculate indicators
        from core import get_indicators
        indicators = get_indicators()
        
        if klines_4h:
            ind = indicators.calculate_all(klines_4h)
        else:
            ind = {}
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'raw_data': {
                'funding_rate': funding,
                'oi_change_4h': oi_change,
                'klines_count': len(klines_4h) if klines_4h else 0,
                'last_kline': klines_4h[-1] if klines_4h else None
            },
            'indicators': {
                'rsi': ind.get('rsi_current'),
                'atr': ind.get('atr_current'),
                'bb_width': ind.get('bb_width_current'),
                'volume_profile': ind.get('volume_profile'),
                'price_range': ind.get('price_range')
            }
        })
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CTR SCANNER ROUTES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @app.route('/ctr')
    def ctr_page():
        """CTR Scanner page"""
        from scheduler.ctr_job import get_ctr_job
        import json
        
        # Get watchlist
        watchlist_str = db.get_setting('ctr_watchlist', '')
        watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        # Get scan results
        scan_results_str = db.get_setting('ctr_last_scan', '[]')
        try:
            scan_results = json.loads(scan_results_str)
        except:
            scan_results = []
        
        # Get last scan time
        last_scan_time = db.get_setting('ctr_last_scan_time', None)
        if last_scan_time:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(last_scan_time.replace('Z', '+00:00'))
                last_scan_time = dt.strftime('%H:%M:%S')
            except:
                pass
        
        # Count statuses
        oversold_count = sum(1 for r in scan_results if r.get('status') == 'Oversold')
        overbought_count = sum(1 for r in scan_results if r.get('status') == 'Overbought')
        
        # Check if running
        ctr_job = get_ctr_job()
        ctr_running = ctr_job.is_running() if ctr_job else False
        
        # Check CTR Only mode
        ctr_only_mode = db.get_setting('ctr_only_mode', '0')
        if isinstance(ctr_only_mode, str):
            ctr_only_mode = ctr_only_mode in ('1', 'true', 'yes')
        
        # Get settings
        settings = {
            'ctr_timeframe': db.get_setting('ctr_timeframe', '15m'),
            'ctr_fast_length': db.get_setting('ctr_fast_length', 21),
            'ctr_slow_length': db.get_setting('ctr_slow_length', 50),
            'ctr_cycle_length': db.get_setting('ctr_cycle_length', 10),
            'ctr_upper': db.get_setting('ctr_upper', 75),
            'ctr_lower': db.get_setting('ctr_lower', 25),
        }
        
        return render_template('ctr.html',
            watchlist=watchlist,
            scan_results=scan_results,
            last_scan_time=last_scan_time,
            oversold_count=oversold_count,
            overbought_count=overbought_count,
            signals_today=0,  # TODO: implement
            ctr_running=ctr_running,
            ctr_only_mode=ctr_only_mode,
            settings=settings
        )
    
    @app.route('/api/ctr/watchlist/add', methods=['POST'])
    def api_ctr_watchlist_add():
        """Add symbol to CTR watchlist"""
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'})
        
        if not symbol.endswith('USDT'):
            return jsonify({'success': False, 'error': 'Symbol must end with USDT'})
        
        # Get current watchlist
        watchlist_str = db.get_setting('ctr_watchlist', '')
        watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        if symbol in watchlist:
            return jsonify({'success': False, 'error': 'Symbol already in watchlist'})
        
        # Validate symbol exists on Binance
        try:
            from core.binance_connector import get_binance_connector
            fetcher = get_binance_connector()
            ticker = fetcher.get_ticker(symbol)
            if not ticker:
                return jsonify({'success': False, 'error': f'Symbol {symbol} not found on Binance'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Validation error: {str(e)}'})
        
        # Add to watchlist
        watchlist.append(symbol)
        db.set_setting('ctr_watchlist', ','.join(watchlist))
        
        return jsonify({'success': True, 'watchlist': watchlist})
    
    @app.route('/api/ctr/watchlist/remove', methods=['POST'])
    def api_ctr_watchlist_remove():
        """Remove symbol from CTR watchlist"""
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'})
        
        # Get current watchlist
        watchlist_str = db.get_setting('ctr_watchlist', '')
        watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        if symbol not in watchlist:
            return jsonify({'success': False, 'error': 'Symbol not in watchlist'})
        
        # Remove from watchlist
        watchlist.remove(symbol)
        db.set_setting('ctr_watchlist', ','.join(watchlist))
        
        return jsonify({'success': True, 'watchlist': watchlist})
    
    @app.route('/api/ctr/settings', methods=['POST'])
    def api_ctr_settings():
        """Save CTR settings"""
        data = request.get_json()
        
        # Save all CTR settings
        ctr_settings = [
            'ctr_timeframe', 'ctr_fast_length', 'ctr_slow_length',
            'ctr_cycle_length', 'ctr_upper', 'ctr_lower'
        ]
        
        for key in ctr_settings:
            if key in data:
                db.set_setting(key, data[key])
        
        # Reload scanner settings
        from detection.ctr_scanner import get_ctr_scanner
        scanner = get_ctr_scanner(db)
        scanner.reload_settings()
        
        return jsonify({'success': True})
    
    @app.route('/api/ctr/start', methods=['POST'])
    def api_ctr_start():
        """Start CTR scanner"""
        from scheduler.ctr_job import start_ctr_job
        
        try:
            job = start_ctr_job(db)
            return jsonify({'success': True, 'running': job.is_running()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/stop', methods=['POST'])
    def api_ctr_stop():
        """Stop CTR scanner"""
        from scheduler.ctr_job import stop_ctr_job
        
        try:
            stop_ctr_job()
            return jsonify({'success': True, 'running': False})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/scan', methods=['POST'])
    def api_ctr_scan():
        """Run CTR scan manually"""
        from detection.ctr_scanner import get_ctr_scanner
        import json
        
        # Get watchlist
        watchlist_str = db.get_setting('ctr_watchlist', '')
        watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        if not watchlist:
            return jsonify({'success': False, 'error': 'Watchlist is empty'})
        
        # Run scan
        scanner = get_ctr_scanner(db)
        results, signals = scanner.scan_watchlist(watchlist)
        
        # Store results
        from datetime import datetime, timezone
        db.set_setting('ctr_last_scan', json.dumps(results))
        db.set_setting('ctr_last_scan_time', datetime.now(timezone.utc).isoformat())
        
        # Send signals to Telegram
        if signals:
            from alerts.telegram_notifier import get_notifier
            notifier = get_notifier()
            for signal in signals:
                message = scanner.format_telegram_signal(signal)
                notifier.send_message(message)
        
        return jsonify({
            'success': True,
            'results_count': len(results),
            'signals_count': len(signals)
        })
    
    @app.route('/api/ctr/only-mode', methods=['POST'])
    def api_ctr_only_mode():
        """Toggle CTR Only mode (disables other scans)"""
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        db.set_setting('ctr_only_mode', '1' if enabled else '0')
        
        return jsonify({'success': True, 'enabled': enabled})
    
    # End of register_api_routes


# Create app with all routes
def get_app():
    """Get configured Flask app"""
    app = create_app()
    register_api_routes(app)
    return app


# Module-level app instance for gunicorn
app = get_app()


# For direct run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
