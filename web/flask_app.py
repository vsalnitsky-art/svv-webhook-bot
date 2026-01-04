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
    
    # Initialize database
    init_db()
    
    # Register routes
    register_routes(app)
    
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
        
        # Count sleepers by state
        sleeper_counts = {
            'total': len(sleepers),
            'watching': len([s for s in sleepers if s.state == 'WATCHING']),
            'building': len([s for s in sleepers if s.state == 'BUILDING']),
            'ready': len([s for s in sleepers if s.state == 'READY'])
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
        
        # Sort by total_score desc
        sleepers.sort(key=lambda x: x.total_score or 0, reverse=True)
        
        return render_template('sleepers.html',
            sleepers=sleepers,
            now=datetime.utcnow()
        )
    
    @app.route('/orderblocks')
    def orderblocks_page():
        """Order blocks page"""
        db = get_db()
        
        # Get active OBs
        from storage.db_models import OrderBlock, OBStatus, get_session
        session = get_session()
        obs = session.query(OrderBlock).filter(
            OrderBlock.status.in_([OBStatus.FRESH.value, OBStatus.TESTED.value])
        ).order_by(OrderBlock.created_at.desc()).limit(50).all()
        session.close()
        
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
                'ready_sleepers': len([s for s in sleepers if s.state == 'READY']),
                'open_trades': len(open_trades),
                'paper_balance': paper_balance
            }
        })
    
    @app.route('/api/sleepers')
    def api_sleepers():
        """Get all sleeper candidates"""
        db = get_db()
        sleepers = db.get_sleepers()
        
        data = []
        for s in sleepers:
            data.append({
                'symbol': s.symbol,
                'state': s.state,
                'direction': s.direction,
                'total_score': round(s.total_score or 0, 1),
                'fuel_score': round(s.fuel_score or 0, 1),
                'volatility_score': round(s.volatility_score or 0, 1),
                'hp': s.hp,
                'funding_rate': s.funding_rate,
                'oi_change_4h': round(s.oi_change_4h or 0, 2),
                'bb_width': round(s.bb_width or 0, 4),
                'rsi': round(s.rsi or 0, 1),
                'updated_at': s.updated_at.isoformat() if s.updated_at else None
            })
        
        return jsonify({'success': True, 'data': data})
    
    @app.route('/api/orderblocks')
    def api_orderblocks():
        """Get active order blocks"""
        from storage.db_models import OrderBlock, OBStatus, get_session
        
        session = get_session()
        obs = session.query(OrderBlock).filter(
            OrderBlock.status.in_([OBStatus.FRESH.value, OBStatus.TESTED.value])
        ).order_by(OrderBlock.created_at.desc()).limit(50).all()
        
        data = []
        for ob in obs:
            data.append({
                'id': ob.id,
                'symbol': ob.symbol,
                'timeframe': ob.timeframe,
                'ob_type': ob.ob_type,
                'ob_high': ob.ob_high,
                'ob_low': ob.ob_low,
                'ob_mid': ob.ob_mid,
                'quality_score': round(ob.quality_score or 0, 1),
                'status': ob.status,
                'touch_count': ob.touch_count,
                'created_at': ob.created_at.isoformat() if ob.created_at else None
            })
        
        session.close()
        return jsonify({'success': True, 'data': data})
    
    @app.route('/api/trades')
    def api_trades():
        """Get recent trades"""
        db = get_db()
        
        limit = request.args.get('limit', 20, type=int)
        trades = db.get_trades(limit=limit)
        
        data = []
        for t in trades:
            data.append({
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size': t.position_size,
                'pnl_usdt': round(t.pnl_usdt or 0, 2),
                'pnl_percent': round(t.pnl_percent or 0, 2),
                'status': t.status,
                'is_paper': t.is_paper,
                'opened_at': t.opened_at.isoformat() if t.opened_at else None,
                'closed_at': t.closed_at.isoformat() if t.closed_at else None
            })
        
        return jsonify({'success': True, 'data': data})
    
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
            
            for key, value in data.items():
                if key in DEFAULT_SETTINGS:
                    db.set_setting(key, value)
            
            db.log_event('INFO', 'SYSTEM', 'Settings updated via API')
            return jsonify({'success': True, 'message': 'Settings updated'})
    
    @app.route('/api/scan/sleepers', methods=['POST'])
    def api_scan_sleepers():
        """Trigger sleeper scan"""
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
        
        data = []
        for e in events:
            data.append({
                'id': e.id,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'level': e.level,
                'category': e.category,
                'message': e.message,
                'symbol': e.symbol
            })
        
        return jsonify({'success': True, 'data': data})
    
    # Register API routes
    register_api_routes.__call__ = lambda: None  # Placeholder


# Create app with all routes
def get_app():
    """Get configured Flask app"""
    app = create_app()
    register_api_routes(app)
    return app


# Module-level app instance
app = None

def get_or_create_app():
    """Get existing app or create new one"""
    global app
    if app is None:
        app = get_app()
    return app

# For direct run
if __name__ == '__main__':
    app = get_or_create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
