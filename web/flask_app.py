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
            'sleeper_building_score': 55,
            'sleeper_ready_score': 70,
            'sleeper_min_volume': 20000000,
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
    
    # ============================================
    # UT BOT MODULE ROUTES
    # ============================================
    
    @app.route('/ut_bot')
    def ut_bot_page():
        """UT Bot Trading page"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            
            return render_template('ut_bot.html',
                status=monitor.get_status(),
                potential_coins=monitor.get_potential_coins(),
                open_trades=monitor.get_open_trades(),
                trade_history=monitor.get_trade_history(limit=20)
            )
        except Exception as e:
            print(f"[UT BOT PAGE] Error: {e}")
            return render_template('ut_bot.html',
                status={
                    'enabled': False,
                    'potential_coins': 0,
                    'open_trades': 0,
                    'top_coin': None,
                    'stats': {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'total_pnl': 0},
                    'config': {'timeframe': '15m', 'atr_period': 10, 'atr_multiplier': 1.0, 'use_heikin_ashi': True}
                },
                potential_coins=[],
                open_trades=[],
                trade_history=[]
            )
    
    @app.route('/api/ut_bot/status')
    def api_ut_bot_status():
        """Get UT Bot module status"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            return jsonify({
                'success': True,
                **monitor.get_status()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/config', methods=['GET', 'POST'])
    def api_ut_bot_config():
        """Get or update UT Bot configuration"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            
            if request.method == 'GET':
                return jsonify({
                    'success': True,
                    'config': monitor.config
                })
            
            # POST - update config
            data = request.get_json()
            if data:
                # Update monitor config
                monitor.update_config(data)
                
                # Save to DB - db.set_setting uses json.dumps internally
                db = get_db()
                for key, value in data.items():
                    db_key = f'ut_bot_{key}'
                    db.set_setting(db_key, value)
                    print(f"[UT BOT] Saved {db_key} = {value}")
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated',
                'config': monitor.config
            })
        except Exception as e:
            print(f"[UT BOT CONFIG ERROR] {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/potential_coins')
    def api_ut_bot_potential_coins():
        """Get potential coins for UT Bot"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            return jsonify({
                'success': True,
                'coins': monitor.get_potential_coins(),
                'count': len(monitor.get_potential_coins())
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/add_coin', methods=['POST'])
    def api_ut_bot_add_coin():
        """Manually add a coin to UT Bot monitoring"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            
            data = request.get_json() or {}
            symbol = data.get('symbol', '').upper()
            direction = data.get('direction', 'LONG').upper()
            score = float(data.get('score', 70.0))
            
            if not symbol:
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            result = monitor.add_coin_manual(symbol, direction, score)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/remove_coin', methods=['POST'])
    def api_ut_bot_remove_coin():
        """Remove a coin from UT Bot monitoring"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            
            data = request.get_json() or {}
            symbol = data.get('symbol', '').upper()
            
            if not symbol:
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            result = monitor.remove_coin(symbol)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/trades')
    def api_ut_bot_trades():
        """Get UT Bot trades (open and history)"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            return jsonify({
                'success': True,
                'open_trades': monitor.get_open_trades(),
                'history': monitor.get_trade_history(limit=50)
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/analyze/<symbol>')
    def api_ut_bot_analyze(symbol):
        """Analyze symbol with UT Bot indicator"""
        try:
            from modules.ut_bot_filter import get_ut_bot_filter
            ut_bot = get_ut_bot_filter()
            
            timeframe = request.args.get('timeframe', '15m')
            signal = ut_bot.analyze(symbol, timeframe=timeframe)
            
            return jsonify({
                'success': True,
                'signal': signal.to_dict()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/check_signals', methods=['POST'])
    def api_ut_bot_check_signals():
        """Manually trigger signal check"""
        try:
            from modules.ut_bot_monitor import get_ut_bot_monitor
            monitor = get_ut_bot_monitor()
            events = monitor.check_signals()
            
            return jsonify({
                'success': True,
                'events': events
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/clear/potential_coins', methods=['POST'])
    def api_ut_bot_clear_potential_coins():
        """Clear all potential coins"""
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            deleted = session.query(UTBotPotentialCoin).delete()
            session.commit()
            session.close()
            print(f"[UT BOT] Cleared {deleted} potential coins")
            return jsonify({'success': True, 'deleted': deleted})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/clear/open_trades', methods=['POST'])
    def api_ut_bot_clear_open_trades():
        """Clear all open trades (mark as CANCELLED)"""
        try:
            from storage.db_models import UTBotPaperTrade, get_session
            from datetime import datetime
            session = get_session()
            open_trades = session.query(UTBotPaperTrade).filter_by(status='OPEN').all()
            count = 0
            for trade in open_trades:
                trade.status = 'CANCELLED'
                trade.closed_at = datetime.now()
                trade.pnl_usdt = 0
                trade.pnl_percent = 0
                count += 1
            session.commit()
            session.close()
            print(f"[UT BOT] Cancelled {count} open trades")
            return jsonify({'success': True, 'cancelled': count})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/clear/trade_history', methods=['POST'])
    def api_ut_bot_clear_trade_history():
        """Clear trade history (delete closed trades)"""
        try:
            from storage.db_models import UTBotPaperTrade, get_session
            session = get_session()
            deleted = session.query(UTBotPaperTrade).filter(
                UTBotPaperTrade.status.in_(['CLOSED', 'CANCELLED'])
            ).delete(synchronize_session='fetch')
            session.commit()
            session.close()
            print(f"[UT BOT] Deleted {deleted} closed trades")
            return jsonify({'success': True, 'deleted': deleted})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ut_bot/clear/all', methods=['POST'])
    def api_ut_bot_clear_all():
        """Clear all UT Bot data"""
        try:
            from storage.db_models import UTBotPotentialCoin, UTBotPaperTrade, get_session
            session = get_session()
            
            coins_deleted = session.query(UTBotPotentialCoin).delete()
            trades_deleted = session.query(UTBotPaperTrade).delete()
            
            session.commit()
            session.close()
            print(f"[UT BOT] Cleared ALL: {coins_deleted} coins, {trades_deleted} trades")
            return jsonify({
                'success': True, 
                'coins_deleted': coins_deleted,
                'trades_deleted': trades_deleted
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
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
