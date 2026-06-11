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
            print(f"[APP] ✓ Bybit Public API working: {len(tickers)} tickers available")
        else:
            print("[APP] ⚠ Bybit API returned empty tickers list")
        
        # Private API check happens in CTR Trade Executor init
        # (it calls session directly to properly detect 401)
        if connector.api_key:
            print(f"[APP]   Bybit API key configured — auth will be tested by Trade Executor")
        else:
            print(f"[APP] ⚠ Bybit API keys not configured — trading disabled")
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
    
    # Auto-start CTR Scanner + Liquidity Map — use before_request to survive Gunicorn fork
    _auto_started = {'ctr': False, 'liq': False, 'funding': False, 'volflow': False, 'coinflow': False, 'exitmon': False, 'whales': False, 'smc': False, 'tm': False, 'top100ob': False, 'liqmap': False, 'apihealth': False}
    
    @app.before_request
    def _maybe_auto_start():
        # Volume Flow — always start
        if not _auto_started['volflow']:
            _auto_started['volflow'] = True
            try:
                from detection.volume_flow import init_volume_flow
                vf_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    vf_tg = get_notifier()
                except:
                    pass
                vf = init_volume_flow(db=get_db(), notifier=vf_tg)
                vf.start()
            except Exception as e:
                print(f"[APP] Failed to start Volume Flow: {e}")
        
        # API Health monitor — always start (lightweight, 6 checks / 2 min)
        if not _auto_started['apihealth']:
            _auto_started['apihealth'] = True
            try:
                from detection.api_health import init_api_health_monitor
                init_api_health_monitor().start()
            except Exception as e:
                print(f"[APP] Failed to start API Health monitor: {e}")
        
        # Liquidity Map
        if not _auto_started['liq']:
            _auto_started['liq'] = True
            try:
                from detection.liquidity_map import init_liquidity_map
                liq = init_liquidity_map(db=get_db())
                liq.start()
            except Exception as e:
                print(f"[APP] Failed to start Liquidity Map: {e}")
        
        # === DEPRECATED — REST-based HeatmapCollector ===
        # Replaced by WebSocket-based OrderBookCollector below. The REST
        # collector was hitting Binance HTTP 418 (datacenter IP blocked)
        # constantly — see logs like "[HEATMAP] BTCUSDT: HTTP 418" — so
        # it was useless. We leave the DB table and its endpoints alone
        # for now (no code depends on them); the worker just stays off.
        # Re-enable here only if Binance unblocks Render IPs OR a proxy
        # is introduced.
        # if not _auto_started.get('heatmap'):
        #     _auto_started['heatmap'] = True
        #     try:
        #         from detection.heatmap_collector import init_heatmap_collector
        #         hm = init_heatmap_collector(db=get_db())
        #         hm.start()
        #     except Exception as e:
        #         print(f"[APP] Failed to start Heatmap Collector: {e}")
        
        # OrderBook Collector — WebSocket-based per-symbol depth20 stream.
        # Lazy: only opens WS connections when the UI requests a symbol.
        # We start the manager itself at boot so the cleanup thread runs;
        # actual subscriptions are demand-driven.
        if not _auto_started.get('orderbook'):
            _auto_started['orderbook'] = True
            try:
                from detection.orderbook_collector import get_orderbook_collector
                obc = get_orderbook_collector()
                obc.start()
            except Exception as e:
                print(f"[APP] Failed to start OrderBook Collector: {e}")
        
        # Funding Rate Monitor
        if not _auto_started['funding']:
            _auto_started['funding'] = True
            try:
                from detection.funding_monitor import init_funding_monitor
                from core.bybit_connector import get_connector
                bybit = get_connector()
                tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    tg = get_notifier()
                except:
                    pass
                fm = init_funding_monitor(bybit_connector=bybit, db=get_db(), notifier=tg)
                fm.start()
            except Exception as e:
                print(f"[APP] Failed to start Funding Monitor: {e}")
        
        # Coin Flow Scanner (depends on Funding Monitor)
        if not _auto_started['coinflow']:
            _auto_started['coinflow'] = True
            try:
                from detection.funding_monitor import get_funding_monitor
                from detection.coin_flow import init_coin_flow
                from detection.signal_validator import init_validator
                fm = get_funding_monitor()
                cf_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    cf_tg = get_notifier()
                except:
                    pass
                cf = init_coin_flow(funding_monitor=fm, notifier=cf_tg)
                cf.start()
                # Signal Validator (no thread, on-demand)
                init_validator(db=get_db(), notifier=cf_tg)
            except Exception as e:
                print(f"[APP] Failed to start Coin Flow: {e}")
        
        # Position Exit Monitor
        if not _auto_started['exitmon']:
            _auto_started['exitmon'] = True
            try:
                from detection.exit_monitor import init_exit_monitor
                from core.bybit_connector import get_connector
                em_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    em_tg = get_notifier()
                except:
                    pass
                bybit = get_connector()
                em = init_exit_monitor(db=get_db(), notifier=em_tg, bybit_connector=bybit)
                em.start()
            except Exception as e:
                print(f"[APP] Failed to start Exit Monitor: {e}")
        
        # Whale Tape — DISABLED (Full Analytics suspended)
        # Market Analytics (Heatmap + Volume Profile + Whale Tape) is stopped.
        # To re-enable, uncomment the block below.
        # if not _auto_started['whales']:
        #     _auto_started['whales'] = True
        #     try:
        #         from detection.whale_tape import init_whale_tape
        #         wt = init_whale_tape(db=get_db())
        #         wt.start()
        #     except Exception as e:
        #         print(f"[APP] Failed to start Whale Tape: {e}")
        _auto_started['whales'] = True  # mark as handled so block is skipped
        
        # Smart Money Scanner
        if not _auto_started['smc']:
            _auto_started['smc'] = True
            try:
                # Initialize forecast engine first — SMC scanner uses it during scans
                from detection.forecast_engine import init_forecast_engine
                init_forecast_engine()
                
                # Momentum Strength module — feeds the Pine-equivalent
                # momStrengthRaw filter into CTR signal evaluation. Without
                # this, calc_ctr falls back to raw STC crossovers and shows
                # signals that wouldn't fire in TradingView.
                try:
                    from detection.market_data import get_market_data
                    from detection.momentum_strength import init_momentum_strength
                    init_momentum_strength(get_market_data())
                except Exception as ms_err:
                    print(f"[APP] MomentumStrength init warning: {ms_err}")
                
                from detection.smc_scanner import init_smc_scanner
                smc_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    smc_tg = get_notifier()
                except:
                    pass
                smc = init_smc_scanner(db=get_db(), notifier=smc_tg)
                smc.start()
            except Exception as e:
                print(f"[APP] Failed to start SMC Scanner: {e}")
        
        # Trade Manager (manages real Bybit positions from SMC signals)
        # Always init so settings/state can be read; actual trading gated by
        # the settings.enabled toggle (default OFF for safety).
        if not _auto_started['tm']:
            _auto_started['tm'] = True
            try:
                from detection.trade_manager import init_trade_manager
                from detection.smc_scanner import get_smc_scanner
                from core.bybit_connector import get_connector
                tm_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    tm_tg = get_notifier()
                except:
                    pass
                tm = init_trade_manager(
                    db=get_db(),
                    notifier=tm_tg,
                    bybit=get_connector(),
                    scanner=get_smc_scanner(),
                )
                tm.start()
            except Exception as e:
                print(f"[APP] Failed to start Trade Manager: {e}")
        
        # Liquidation Map daemon — tracks estimated liquidation clusters
        # for BTC + ETH (background) plus any on-demand symbol requested
        # via the dashboard. Tier 2 architecture: Binance/Bybit aggregated
        # OI estimation + Hyperliquid real position overlay.
        #
        # Always auto-starts on boot. User can toggle off at runtime via UI
        # to STOP the daemon, but next deploy/restart it comes back on. This
        # is intentional — the module accumulates data continuously and
        # needs to run unless explicitly disabled in this session.
        if not _auto_started['liqmap']:
            _auto_started['liqmap'] = True
            try:
                db = get_db()
                # Force setting to '1' so UI reflects enabled state too.
                # Daemon is always started regardless of previous setting.
                db.set_setting('liquidation_map_enabled', '1')
                from detection.liquidation_map import init_liquidation_map
                from detection.market_data import get_market_data
                lm = init_liquidation_map(
                    db=db,
                    market_data=get_market_data(),
                )
                lm.start()
                print("[APP] Liquidation Map auto-started (default ON)")
            except Exception as e:
                print(f"[APP] Failed to start Liquidation Map: {e}")
        
        # Liquidity Map Signal Scanner — watchlist-wide composite scoring
        # (forecast + imbalance + liq fuel + squeeze + manipulation) with
        # Telegram alerts. The daemon thread always runs; whether it
        # actually scans is gated by the liqmap_signal_enabled DB setting
        # (default OFF — operator opts in from the dashboard).
        if not _auto_started.get('liqsig'):
            _auto_started['liqsig'] = True
            try:
                from detection.liqmap_signal_scanner import init_liqmap_signal_scanner
                from alerts.telegram_notifier import get_notifier
                init_liqmap_signal_scanner(db=get_db(), notifier=get_notifier())
                print("[APP] Liqmap Signal Scanner thread started "
                      "(scanning gated by liqmap_signal_enabled)")
            except Exception as e:
                print(f"[APP] Failed to start Liqmap Signal Scanner: {e}")
        
        # CTR Scanner — only if auto-start enabled
        if not _auto_started['ctr']:
            _auto_started['ctr'] = True
            try:
                db = get_db()
                if db.get_setting('ctr_auto_start', '0') == '1':
                    import threading
                    def _do_start():
                        import time
                        time.sleep(5)
                        try:
                            from scheduler.ctr_job import start_ctr_job
                            job = start_ctr_job(get_db())
                            if job.is_running():
                                print("[CTR Job] ✅ Auto-started after server restart")
                            else:
                                print("[CTR Job] ⚠️ Auto-start failed — scanner not running")
                        except Exception as e:
                            print(f"[CTR Job] ❌ Auto-start error: {e}")
                            import traceback
                            traceback.print_exc()
                    threading.Thread(target=_do_start, daemon=True).start()
                    print("[APP] CTR Scanner auto-start scheduled (worker process)")
            except Exception as e:
                print(f"[APP] CTR auto-start check failed: {e}")
        
        # TOP-100 4H OB Scanner — Variant B (scheduled scan, informational)
        # Always init the scanner so settings can be read via API. Whether
        # the scheduler thread actually starts is controlled by the
        # `top100_ob_enabled` setting (default OFF — user opts in).
        if not _auto_started['top100ob']:
            _auto_started['top100ob'] = True
            try:
                from detection.top100_ob_scanner import get_top100_ob_scanner
                top100_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    top100_tg = get_notifier()
                except Exception:
                    pass
                scanner = get_top100_ob_scanner(telegram_notifier=top100_tg)
                # Restore persisted settings from DB so user's choices
                # survive worker restart on Render. We mirror only the
                # toggleable bits — schedule/throttle/timeframe stay code
                # constants since they don't need to be user-tunable yet.
                try:
                    db = get_db()
                    enabled = db.get_setting('top100_ob_enabled', '0') == '1'
                    tg_alerts = db.get_setting('top100_ob_telegram', '1') == '1'
                    include_bos = db.get_setting('top100_ob_include_bos', '0') == '1'
                    min_vol_str = db.get_setting('top100_ob_min_vol_usd', '100000000')
                    try:
                        min_vol = float(min_vol_str)
                    except (TypeError, ValueError):
                        min_vol = 100_000_000
                    # New tunables: TF, scan interval, fresh-window, zone
                    # thresholds. All persisted as strings, parsed to the
                    # right type with safe-default fallbacks. Whitelisted
                    # values are also re-validated inside update_settings()
                    # so a corrupt DB row can't poison the scanner state.
                    timeframe = db.get_setting('top100_ob_timeframe', '1h')
                    try:
                        scan_interval_min = int(db.get_setting(
                            'top100_ob_scan_interval_min', '10'))
                    except (TypeError, ValueError):
                        scan_interval_min = 10
                    try:
                        fresh_hours = float(db.get_setting(
                            'top100_ob_fresh_window_hours', '3'))
                    except (TypeError, ValueError):
                        fresh_hours = 3.0
                    try:
                        long_max = float(db.get_setting(
                            'top100_ob_long_max_pct', '20'))
                    except (TypeError, ValueError):
                        long_max = 20.0
                    try:
                        short_min = float(db.get_setting(
                            'top100_ob_short_min_pct', '80'))
                    except (TypeError, ValueError):
                        short_min = 80.0
                    scanner.update_settings(
                        enabled=enabled,
                        telegram_alerts=tg_alerts,
                        include_bos_alerts=include_bos,
                        min_quote_volume_usd=min_vol,
                        timeframe=timeframe,
                        scan_interval_min=scan_interval_min,
                        fresh_window_hours=fresh_hours,
                        long_max_pct=long_max,
                        short_min_pct=short_min,
                    )
                    if enabled:
                        scanner.start()
                        print("[APP] TOP-100 OB Scanner auto-started")
                    else:
                        print("[APP] TOP-100 OB Scanner initialized (disabled by setting)")
                except Exception as e:
                    print(f"[APP] TOP-100 OB Scanner settings restore failed: {e}")
            except Exception as e:
                print(f"[APP] Failed to init TOP-100 OB Scanner: {e}")
        
        # Volumized OB Radar — same auto-start pattern as TOP-100 but a
        # different scanner (Pine Volumized algorithm + P/D zone filter).
        # Settings live as a single JSON blob under 'volumized_radar_settings',
        # loaded inside the singleton constructor — so all we do here is
        # call get_volumized_ob_radar() (constructs + auto-loads) and start
        # the daemon thread if `enabled=True` in those settings.
        _auto_started.setdefault('volradar', False)
        if not _auto_started['volradar']:
            _auto_started['volradar'] = True
            try:
                from detection.volumized_ob_radar import get_volumized_ob_radar
                vol_tg = None
                try:
                    from alerts.telegram_notifier import get_notifier
                    vol_tg = get_notifier()
                except Exception:
                    pass
                radar = get_volumized_ob_radar(telegram_notifier=vol_tg)
                if radar.get_settings().get('enabled'):
                    radar.start()
                    print("[APP] Volumized OB Radar auto-started")
                else:
                    print("[APP] Volumized OB Radar initialized (disabled by setting)")
            except Exception as e:
                print(f"[APP] Failed to init Volumized OB Radar: {e}")
    
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
        from storage.db_operations import get_db
        import json
        
        db = get_db()
        
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
        
        # Check if running and get status
        try:
            ctr_job = get_ctr_job(db)
            ctr_running = ctr_job.is_running()
            ctr_status = ctr_job.get_status()
            ws_connected = ctr_status.get('ws_connected', False)
            stats = ctr_status.get('stats', {})
            
            # Enrich scan_results with ZLT trend data (for Jinja template)
            if hasattr(ctr_job, '_zl_service') and ctr_job._zl_service and ctr_job._zl_service.enabled:
                for r in scan_results:
                    sym = r.get('symbol', '')
                    zl_data = ctr_job._zl_service._trends.get(sym, {})
                    zl_summary = {}
                    for key, label in [('5', '5m'), ('15', '15m'), ('60', '1h'), ('240', '4h')]:
                        td = zl_data.get(key)
                        if td:
                            zl_summary[label] = td['trend'].upper()
                    r['zl_trend'] = zl_summary if zl_summary else None
        except:
            ctr_running = False
            ws_connected = False
            stats = {}
        
        # Check CTR Only mode
        ctr_only_mode = db.get_setting('ctr_only_mode', '0')
        if isinstance(ctr_only_mode, str):
            ctr_only_mode = ctr_only_mode in ('1', 'true', 'yes')
        
        # Get settings
        smc_enabled_str = db.get_setting('ctr_smc_filter_enabled', '0')
        smc_filter_enabled = smc_enabled_str in ('1', 'true', 'True', 'yes')
        
        settings = {
            'ctr_timeframe': db.get_setting('ctr_timeframe', '15m'),
            'ctr_fast_length': db.get_setting('ctr_fast_length', 21),
            'ctr_slow_length': db.get_setting('ctr_slow_length', 50),
            'ctr_cycle_length': db.get_setting('ctr_cycle_length', 10),
            'ctr_upper': db.get_setting('ctr_upper', 75),
            'ctr_lower': db.get_setting('ctr_lower', 25),
            # Optional signal filters
            'ctr_use_trend_guard': db.get_setting('ctr_use_trend_guard', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_use_gap_detection': db.get_setting('ctr_use_gap_detection', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_use_cooldown': db.get_setting('ctr_use_cooldown', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_cooldown_seconds': db.get_setting('ctr_cooldown_seconds', 300),
            # SMC Filter
            'ctr_smc_filter_enabled': smc_filter_enabled,
            'ctr_smc_swing_length': db.get_setting('ctr_smc_swing_length', 50),
            'ctr_smc_zone_threshold': db.get_setting('ctr_smc_zone_threshold', 1.0),
            'ctr_smc_require_trend': db.get_setting('ctr_smc_require_trend', '1') in ('1', 'true', 'True', 'yes'),
            # SMC Trend Filter (HTF direction)
            'ctr_smc_trend_enabled': db.get_setting('ctr_smc_trend_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_smc_trend_swing_4h': db.get_setting('ctr_smc_trend_swing_4h', 50),
            'ctr_smc_trend_swing_1h': db.get_setting('ctr_smc_trend_swing_1h', 50),
            'ctr_smc_trend_mode': db.get_setting('ctr_smc_trend_mode', 'both'),
            'ctr_smc_trend_refresh': db.get_setting('ctr_smc_trend_refresh', 900),
            'ctr_smc_trend_block_neutral': db.get_setting('ctr_smc_trend_block_neutral', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_smc_trend_early_warning': db.get_setting('ctr_smc_trend_early_warning', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_smc_trend_swing_15m': db.get_setting('ctr_smc_trend_swing_15m', 20),
            'ctr_telegram_mode': db.get_setting('ctr_telegram_mode', 'all'),
            # SL Monitor
            'ctr_sl_monitor_enabled': db.get_setting('ctr_sl_monitor_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_sl_monitor_pct': db.get_setting('ctr_sl_monitor_pct', '0'),
            'ctr_sl_check_interval': db.get_setting('ctr_sl_check_interval', '5'),
            # FVG Detector
            'ctr_fvg_enabled': db.get_setting('ctr_fvg_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_timeframe': db.get_setting('ctr_fvg_timeframe', '15m'),
            'ctr_fvg_min_pct': db.get_setting('ctr_fvg_min_pct', '0.1'),
            'ctr_fvg_max_per_symbol': db.get_setting('ctr_fvg_max_per_symbol', '2'),
            'ctr_fvg_rr_ratio': db.get_setting('ctr_fvg_rr_ratio', '1.5'),
            'ctr_fvg_sl_buffer_pct': db.get_setting('ctr_fvg_sl_buffer_pct', '0.2'),
            'ctr_fvg_scan_interval': db.get_setting('ctr_fvg_scan_interval', '60'),
            'ctr_fvg_check_interval': db.get_setting('ctr_fvg_check_interval', '3'),
            'ctr_fvg_trend_filter': db.get_setting('ctr_fvg_trend_filter', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_trend_fast_ema': db.get_setting('ctr_fvg_trend_fast_ema', '5'),
            'ctr_fvg_trend_slow_ema': db.get_setting('ctr_fvg_trend_slow_ema', '13'),
            'ctr_fvg_htf_trend': db.get_setting('ctr_fvg_htf_trend', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_htf_timeframe': db.get_setting('ctr_fvg_htf_timeframe', '1h'),
            'ctr_fvg_htf_fast_ema': db.get_setting('ctr_fvg_htf_fast_ema', '8'),
            'ctr_fvg_htf_slow_ema': db.get_setting('ctr_fvg_htf_slow_ema', '21'),
            'ctr_fvg_retest_enabled': db.get_setting('ctr_fvg_retest_enabled', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_instant_enabled': db.get_setting('ctr_fvg_instant_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_zl_trend': db.get_setting('ctr_fvg_zl_trend', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_zl_15m': db.get_setting('ctr_fvg_zl_15m', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_zl_1h': db.get_setting('ctr_fvg_zl_1h', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_zl_4h': db.get_setting('ctr_fvg_zl_4h', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_fvg_zl_length': db.get_setting('ctr_fvg_zl_length', '70'),
            'ctr_fvg_zl_mult': db.get_setting('ctr_fvg_zl_mult', '1.2'),
            'ctr_fvg_zl_5m': db.get_setting('ctr_fvg_zl_5m', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_zl_bot_enabled': db.get_setting('ctr_zl_bot_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_zl_bot_partial_pct': db.get_setting('ctr_zl_bot_partial_pct', '50'),
            # CTR Fast Scanner toggle + EMA Trend Filter
            'ctr_scanner_enabled': db.get_setting('ctr_scanner_enabled', '1') in ('1', 'true', 'True', 'yes'),
            'ctr_ema_trend_enabled': db.get_setting('ctr_ema_trend_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'ctr_ema_trend_fast': db.get_setting('ctr_ema_trend_fast', '5'),
            'ctr_ema_trend_slow': db.get_setting('ctr_ema_trend_slow', '13'),
            # FVG TP Manager
            'fvg_tp_manager_enabled': db.get_setting('fvg_tp_manager_enabled', '0') in ('1', 'true', 'True', 'yes'),
            'fvg_tp_trigger_pct': db.get_setting('fvg_tp_trigger_pct', '0.5'),
            'fvg_tp_close_pct': db.get_setting('fvg_tp_close_pct', '50'),
            'fvg_tp_be_buffer_pct': db.get_setting('fvg_tp_be_buffer_pct', '0.05'),
            'fvg_tp_trail_pct': db.get_setting('fvg_tp_trail_pct', '0.3'),
            'fvg_tp_trail_start_pct': db.get_setting('fvg_tp_trail_start_pct', '0.8'),
        }
        
        # Статистика фільтрації
        signals_filtered = stats.get('signals_filtered', 0)
        
        # Get executed signals history
        signals_str = db.get_setting('ctr_signals', '[]')
        try:
            ctr_signals = json.loads(signals_str)
            # Sort by timestamp descending (newest first)
            ctr_signals = sorted(ctr_signals, key=lambda x: x.get('timestamp', ''), reverse=True)
        except:
            ctr_signals = []
        
        # Trade settings
        trade_settings = {
            'enabled': False,
            'leverage': 10,
            'deposit_pct': 5,
            'sizing_mode': 'percent',
            'fixed_margin': 10,
            'tp_pct': 0,
            'sl_pct': 0,
            'max_positions': 5,
            'trade_symbols': [],
        }
        try:
            trade_settings = ctr_job.get_trade_settings()
        except:
            pass
        
        return render_template('ctr.html',
            watchlist=watchlist,
            scan_results=scan_results,
            last_scan_time=last_scan_time,
            oversold_count=oversold_count,
            overbought_count=overbought_count,
            signals_today=stats.get('signals_sent', 0),
            signals_filtered=signals_filtered,
            ctr_running=ctr_running,
            ws_connected=ws_connected,
            smc_filter_enabled=smc_filter_enabled,
            ctr_only_mode=ctr_only_mode,
            settings=settings,
            stats=stats,
            ctr_signals=ctr_signals,
            trade_settings=trade_settings
        )
    
    @app.route('/api/ctr/watchlist/add', methods=['POST'])
    def api_ctr_watchlist_add():
        """Add symbol to CTR watchlist"""
        db = get_db()
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
        db = get_db()
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
        db = get_db()
        data = request.get_json()
        
        # Save all CTR settings
        ctr_settings = [
            'ctr_timeframe', 'ctr_fast_length', 'ctr_slow_length',
            'ctr_cycle_length', 'ctr_upper', 'ctr_lower',
            # Optional signal filters
            'ctr_use_trend_guard', 'ctr_use_gap_detection',
            'ctr_use_cooldown', 'ctr_cooldown_seconds',
            # SMC Filter settings
            'ctr_smc_filter_enabled', 'ctr_smc_swing_length', 'ctr_smc_zone_threshold',
            'ctr_smc_require_trend',
            # SMC Trend Filter (HTF direction)
            'ctr_smc_trend_enabled', 'ctr_smc_trend_swing_4h', 'ctr_smc_trend_swing_1h',
            'ctr_smc_trend_mode', 'ctr_smc_trend_refresh', 'ctr_smc_trend_block_neutral',
            'ctr_smc_trend_early_warning', 'ctr_smc_trend_swing_15m',
            # Telegram mode
            'ctr_telegram_mode',
            # SL Monitor
            'ctr_sl_monitor_enabled', 'ctr_sl_monitor_pct', 'ctr_sl_check_interval',
            # FVG Detector
            'ctr_fvg_enabled', 'ctr_fvg_timeframe', 'ctr_fvg_min_pct',
            'ctr_fvg_max_per_symbol', 'ctr_fvg_rr_ratio', 'ctr_fvg_sl_buffer_pct',
            'ctr_fvg_scan_interval', 'ctr_fvg_check_interval', 'ctr_fvg_trend_filter',
            'ctr_fvg_trend_fast_ema', 'ctr_fvg_trend_slow_ema',
            'ctr_fvg_htf_trend', 'ctr_fvg_htf_timeframe',
            'ctr_fvg_htf_fast_ema', 'ctr_fvg_htf_slow_ema',
            'ctr_fvg_retest_enabled', 'ctr_fvg_instant_enabled',
            'ctr_fvg_zl_trend', 'ctr_fvg_zl_5m', 'ctr_fvg_zl_15m', 'ctr_fvg_zl_1h', 'ctr_fvg_zl_4h',
            'ctr_fvg_zl_length', 'ctr_fvg_zl_mult',
            'ctr_zl_bot_enabled', 'ctr_zl_bot_partial_pct',
            # CTR Fast Scanner toggle + EMA Trend Filter
            'ctr_scanner_enabled', 'ctr_ema_trend_enabled',
            'ctr_ema_trend_fast', 'ctr_ema_trend_slow',
            # FVG TP Manager
            'fvg_tp_manager_enabled', 'fvg_tp_trigger_pct',
            'fvg_tp_close_pct', 'fvg_tp_be_buffer_pct',
            'fvg_tp_trail_pct', 'fvg_tp_trail_start_pct',
        ]
        
        for key in ctr_settings:
            if key in data:
                db.set_setting(key, str(data[key]))
        
        # Reload scanner settings (if running)
        from scheduler.ctr_job import get_ctr_job
        try:
            job = get_ctr_job(db)
            job.reload_settings()
        except:
            pass  # Scanner not initialized yet
        
        return jsonify({'success': True})
    
    @app.route('/api/ctr/start', methods=['POST'])
    def api_ctr_start():
        """Start CTR scanner"""
        from scheduler.ctr_job import start_ctr_job
        db = get_db()
        
        try:
            job = start_ctr_job(db)
            # Remember state for auto-start after restart
            db.set_setting('ctr_auto_start', '1')
            return jsonify({'success': True, 'running': job.is_running()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/stop', methods=['POST'])
    def api_ctr_stop():
        """Stop CTR scanner"""
        from scheduler.ctr_job import stop_ctr_job
        
        try:
            stop_ctr_job()
            # Clear auto-start flag
            db = get_db()
            db.set_setting('ctr_auto_start', '0')
            return jsonify({'success': True, 'running': False})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/scan', methods=['POST'])
    def api_ctr_scan():
        """Run CTR scan manually"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        
        job = get_ctr_job(db)
        
        if not job.is_running():
            return jsonify({'success': False, 'error': 'CTR Scanner is not running. Start it first.'})
        
        # Run scan
        results = job.scan_now()
        
        return jsonify({
            'success': True,
            'results_count': len(results),
            'signals_count': 0  # Signals are sent automatically
        })
    
    @app.route('/api/ctr/smc-trend', methods=['GET'])
    def api_ctr_smc_trend():
        """Get SMC Trend Filter status for all symbols"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        
        try:
            job = get_ctr_job(db)
            return jsonify(job.get_smc_trend_status())
        except Exception as e:
            return jsonify({'enabled': False, 'error': str(e)})
    
    @app.route('/api/ctr/signals/delete', methods=['POST'])
    def api_ctr_signal_delete():
        """Delete a specific CTR signal by timestamp"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        
        data = request.get_json()
        timestamp = data.get('timestamp')
        
        if not timestamp:
            return jsonify({'success': False, 'error': 'Missing signal timestamp'})
        
        try:
            job = get_ctr_job(db)
            success = job.delete_signal(timestamp)
            return jsonify({'success': success})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/signals/clear', methods=['POST'])
    def api_ctr_signals_clear():
        """Clear all CTR signals"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        
        try:
            job = get_ctr_job(db)
            count = job.clear_signals()
            return jsonify({'success': True, 'cleared': count})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/ctr/only-mode', methods=['POST'])
    def api_ctr_only_mode():
        """Toggle CTR Only mode (disables other scans)"""
        db = get_db()
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        db.set_setting('ctr_only_mode', '1' if enabled else '0')
        
        return jsonify({'success': True, 'enabled': enabled})
    
    # === CTR TRADE API ===
    
    @app.route('/api/ctr/trade/settings', methods=['GET'])
    def api_ctr_trade_settings_get():
        """Отримати налаштування торгівлі"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        return jsonify(job.get_trade_settings())
    
    @app.route('/api/ctr/trade/settings', methods=['POST'])
    def api_ctr_trade_settings_save():
        """Зберегти налаштування торгівлі"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        data = request.get_json()
        
        success = job.save_trade_settings(data)
        return jsonify({'success': success, 'settings': job.get_trade_settings()})
    
    @app.route('/api/ctr/trade/toggle', methods=['POST'])
    def api_ctr_trade_toggle():
        """Увімкнути/вимкнути Auto-Trade"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        data = request.get_json()
        
        enabled = data.get('enabled', False)
        success = job.save_trade_settings({'enabled': enabled})
        
        status = "ENABLED ✅" if enabled else "DISABLED ❌"
        print(f"[CTR Trade] Auto-Trade {status}")
        
        return jsonify({'success': success, 'enabled': enabled})
    
    @app.route('/api/ctr/trade/symbol', methods=['POST'])
    def api_ctr_trade_symbol():
        """Позначити/зняти символ для торгівлі"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        enabled = data.get('enabled', False)
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'})
        
        trade_symbols = job.toggle_trade_symbol(symbol, enabled)
        return jsonify({'success': True, 'symbol': symbol, 'enabled': enabled,
                        'trade_symbols': trade_symbols})
    
    @app.route('/api/ctr/trade/status', methods=['GET'])
    def api_ctr_trade_status():
        """Статус торгівлі: баланс, позиції"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        return jsonify(job.get_trade_status())
    
    @app.route('/api/ctr/trade/log', methods=['GET'])
    def api_ctr_trade_log():
        """Історія торгів"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        limit = request.args.get('limit', 50, type=int)
        return jsonify({'log': job.get_trade_log(limit)})
    
    @app.route('/api/ctr/positions', methods=['GET'])
    def api_ctr_positions():
        """Bybit exchange positions only"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        
        result = {
            'positions': [],
            'bybit_connected': False,
            'auth_error': '',
            'balance': 0,
            'max_positions': 5,
            'sl_monitor_enabled': db.get_setting('ctr_sl_monitor_enabled', '0') in ('1', 'true', 'True'),
            'sl_monitor_pct': float(db.get_setting('ctr_sl_monitor_pct', '0')),
        }
        
        try:
            trade_status = job.get_trade_status()
            if trade_status.get('available'):
                result['bybit_connected'] = trade_status.get('auth_ok', False)
                result['auth_error'] = trade_status.get('error', '')
                result['balance'] = trade_status.get('balance', 0)
                result['max_positions'] = trade_status.get('max_positions', 5)
                if trade_status.get('auth_ok'):
                    result['positions'] = trade_status.get('positions', [])
        except:
            pass
        
        return jsonify(result)
    
    @app.route('/api/ctr/trade/reconnect', methods=['POST'])
    def api_ctr_trade_reconnect():
        """Reconnect to Bybit API"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        
        try:
            if hasattr(job, '_trade_executor') and job._trade_executor:
                result = job._trade_executor.reconnect()
                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': 'Trade executor not initialized'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ========================================
    # FVG Detector Routes
    # ========================================
    
    @app.route('/api/ctr/fvg/zones', methods=['GET'])
    def api_ctr_fvg_zones():
        """Get all FVG zones"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        return jsonify({
            'zones': job.get_fvg_zones(),
            'stats': job.get_fvg_stats(),
        })
    
    @app.route('/api/ctr/fvg/clear', methods=['POST'])
    def api_ctr_fvg_clear():
        """Clear all FVG zones"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        count = job.clear_fvg_zones()
        return jsonify({'success': True, 'cleared': count})
    
    @app.route('/api/ctr/fvg/scan', methods=['POST'])
    def api_ctr_fvg_scan():
        """Manual FVG scan"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        job.scan_fvg_now()
        return jsonify({'success': True})
    
    @app.route('/api/ctr/zl-bot/states', methods=['GET'])
    def api_ctr_zl_bot_states():
        """Get ZLT Bot states for all symbols"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        if hasattr(job, '_zl_bot') and job._zl_bot:
            return jsonify(job._zl_bot.get_stats())
        return jsonify({'enabled': False, 'states': {}})
    
    @app.route('/api/ctr/zl-bot/reset', methods=['POST'])
    def api_ctr_zl_bot_reset():
        """Reset ZLT Bot states"""
        from scheduler.ctr_job import get_ctr_job
        db = get_db()
        job = get_ctr_job(db)
        if hasattr(job, '_zl_bot') and job._zl_bot:
            job._zl_bot.reset_all()
        # Also clear persisted state
        db.set_setting('zl_bot_states', {})
        return jsonify({'status': 'ok'})
    
    # ========================================
    # BTC Liquidity Map Routes
    # ========================================
    
    @app.route('/api/liquidity/current')
    def api_liquidity_current():
        """Current liquidity walls."""
        from detection.liquidity_map import get_liquidity_map
        lm = get_liquidity_map()
        if not lm:
            return jsonify({'error': 'Liquidity Map not running'}), 503
        return jsonify(lm.get_current())
    
    @app.route('/api/liquidity/persistent')
    def api_liquidity_persistent():
        """Persistent walls (institutional levels)."""
        from detection.liquidity_map import get_liquidity_map
        lm = get_liquidity_map()
        if not lm:
            return jsonify({'error': 'Liquidity Map not running'}), 503
        return jsonify(lm.get_persistent_walls())
    
    @app.route('/api/liquidity/summary')
    def api_liquidity_summary():
        """Compact summary for dashboard."""
        from detection.liquidity_map import get_liquidity_map
        lm = get_liquidity_map()
        if not lm:
            return jsonify({'running': False})
        return jsonify(lm.get_summary())
    
    @app.route('/api/liquidity/bias-history')
    def api_liquidity_bias_history():
        """Bias history for chart. ?date=2026-04-05 (local date from browser)."""
        from detection.liquidity_map import HISTORY_DAYS
        db = get_db()
        
        req_date = request.args.get('date', '')
        
        # Scan today + past days + tomorrow (covers timezone differences)
        available = []
        for i in range(-1, HISTORY_DAYS + 1):
            d = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
            key = f'liq_bias_{d}'
            data = db.get_setting(key, [])
            if isinstance(data, list) and len(data) > 0:
                available.append({'date': d, 'points': len(data)})
        
        # Sort by date descending (newest first)
        available.sort(key=lambda x: x['date'], reverse=True)
        
        # If no data for requested date, use latest available
        data = db.get_setting(f'liq_bias_{req_date}', [])
        if (not isinstance(data, list) or len(data) == 0) and available:
            req_date = available[0]['date']
            data = db.get_setting(f'liq_bias_{req_date}', [])
            if not isinstance(data, list):
                data = []
        
        return jsonify({
            'date': req_date,
            'data': data,
            'available_days': available,
        })
    
    # ========================================
    # Market Analytics Routes (Stage 1: Heatmap)
    # ========================================
    
    @app.route('/market-analytics')
    def market_analytics_page():
        """Market Microstructure Analytics page with tabs."""
        return render_template('market_analytics.html')
    
    @app.route('/api/heatmap/liquidity')
    def api_heatmap_liquidity():
        """Liquidity Heatmap data — multi-symbol.

        Query params:
          symbol: default BTCUSDT
          hours:  int — time window (default 24, max 168 = 7d)
        """
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        try:
            hours = int(request.args.get('hours', 24))
            hours = max(1, min(hours, 168))
        except:
            hours = 24

        # Prefer new multi-symbol HeatmapCollector. Fall back to legacy
        # BTC-only LiquidityMap when the collector hasn't started yet
        # (mostly for the first request after a cold deploy).
        from detection.heatmap_collector import get_heatmap_collector
        hc = get_heatmap_collector()
        if hc:
            return jsonify(hc.get_heatmap_data(symbol=symbol, hours=hours))

        from detection.liquidity_map import get_liquidity_map
        lm = get_liquidity_map()
        if not lm:
            return jsonify({'error': 'Heatmap not initialized', 'rows': []})
        return jsonify(lm.get_heatmap_data(hours=hours))

    @app.route('/api/heatmap/symbols')
    def api_heatmap_symbols():
        """List symbols currently tracked by HeatmapCollector, plus per-symbol
        last-seen status and the union of symbols that have any stored data.
        """
        from detection.heatmap_collector import get_heatmap_collector
        hc = get_heatmap_collector()
        configured = []
        status = {}
        if hc:
            st = hc.get_status()
            configured = st.get('symbols', [])
            status = st.get('last_results', {})

        stored = []
        try:
            stored = get_db().get_liq_heatmap_symbols()
        except Exception:
            pass

        # Union: configured first (preserve order), then any stored not yet listed
        seen = set(configured)
        union = list(configured)
        for s in stored:
            if s not in seen:
                seen.add(s)
                union.append(s)

        return jsonify({
            'configured': configured,
            'stored': stored,
            'all': union,
            'status': status,
        })

    @app.route('/api/heatmap/symbols/set', methods=['POST'])
    def api_heatmap_symbols_set():
        """Replace the tracked symbol list. Body: {symbols: ["BTCUSDT","ETHUSDT",...]}"""
        from detection.heatmap_collector import get_heatmap_collector
        hc = get_heatmap_collector()
        if not hc:
            return jsonify({'ok': False, 'reason': 'collector not initialized'})
        data = request.get_json() or {}
        items = data.get('symbols', []) or []
        if not isinstance(items, list):
            return jsonify({'ok': False, 'reason': 'symbols must be a list'})
        applied = hc.set_symbols(items)
        return jsonify({'ok': True, 'symbols': applied})

    @app.route('/api/heatmap/candles')
    def api_heatmap_candles():
        """Candles for the heatmap overlay.

        Query params:
          symbol:   default BTCUSDT
          interval: 1m|5m|15m|30m|1h|4h  (default 15m)
          hours:    1..168 (default 24)
        """
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        interval = request.args.get('interval', '15m')
        try:
            hours = int(request.args.get('hours', 24))
            hours = max(1, min(hours, 168))
        except:
            hours = 24

        # Pick a sensible kline count for the chosen interval+window
        per_hour = {'1m': 60, '5m': 12, '15m': 4, '30m': 2, '1h': 1, '4h': 0.25}
        limit = int(min(1000, max(50, hours * per_hour.get(interval, 4))))

        try:
            from detection.market_data import get_market_data
            md = get_market_data()
            if md is None:
                return jsonify({'ok': False, 'reason': 'market_data unavailable',
                                 'candles': []})
            klines = md.fetch_klines(symbol, interval=interval, limit=limit)
            if not klines:
                return jsonify({'ok': True, 'candles': [], 'symbol': symbol,
                                 'interval': interval})
            # Normalize to {t,o,h,l,c} expected by the renderer
            candles = []
            for k in klines:
                t = k.get('t') or k.get('time') or k.get('open_time') or 0
                o = float(k.get('o') or k.get('open') or 0)
                h = float(k.get('h') or k.get('high') or 0)
                l = float(k.get('l') or k.get('low') or 0)
                c = float(k.get('c') or k.get('close') or k.get('p') or 0)
                if c == 0:
                    continue
                candles.append({'t': int(t), 'o': o, 'h': h, 'l': l, 'c': c})
            return jsonify({'ok': True, 'symbol': symbol, 'interval': interval,
                             'candles': candles})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e), 'candles': []})
    
    @app.route('/api/volume-profile')
    def api_volume_profile():
        """Volume Profile (POC / VAH / VAL) for a symbol.
        
        Query params:
          symbol: default BTCUSDT
          hours: 1-168 (default 24). Note: Binance 1m klines max ~25h per request.
          buckets: 20-100 (default 50)
        """
        from detection.volume_profile import build_volume_profile
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        try:
            hours = int(request.args.get('hours', 24))
            hours = max(1, min(hours, 168))
        except:
            hours = 24
        try:
            buckets = int(request.args.get('buckets', 50))
            buckets = max(20, min(buckets, 100))
        except:
            buckets = 50
        
        return jsonify(build_volume_profile(symbol, hours=hours, buckets=buckets))
    
    @app.route('/api/whales/state')
    def api_whales_state():
        """Whale Tape state: recent large trades + stats.
        
        Query params:
          limit: max trades to return (default 100)
          window: stats window in minutes (default 60)
        """
        from detection.whale_tape import get_whale_tape
        wt = get_whale_tape()
        if not wt:
            return jsonify({'trades': [], 'stats': {}, 'running': False, 'error': 'Not initialized'})
        
        try:
            limit = int(request.args.get('limit', 100))
            limit = max(10, min(limit, 500))
        except:
            limit = 100
        try:
            window = int(request.args.get('window', 60))
            window = max(1, min(window, 1440))
        except:
            window = 60
        
        return jsonify(wt.get_state(limit=limit, window_minutes=window))
    
    @app.route('/api/whales/threshold', methods=['GET', 'POST'])
    def api_whales_threshold():
        """Get or set minimum trade size threshold (USD)."""
        from detection.whale_tape import get_whale_tape
        wt = get_whale_tape()
        if not wt:
            return jsonify({'ok': False, 'error': 'Not initialized'})
        
        if request.method == 'GET':
            return jsonify({'ok': True, 'threshold': wt.get_threshold()})
        
        data = request.get_json() or {}
        try:
            usd = int(data.get('threshold', 100000))
        except:
            return jsonify({'ok': False, 'error': 'Invalid threshold'})
        
        ok = wt.set_threshold(usd)
        return jsonify({'ok': ok, 'threshold': wt.get_threshold()})
    
    # ========================================
    # Position Exit Monitor Routes
    # ========================================
    
    @app.route('/api/exitmon/state')
    def api_exitmon_state():
        """Current Exit Monitor state for all tracked positions."""
        from detection.exit_monitor import get_exit_monitor
        em = get_exit_monitor()
        if not em:
            return jsonify({'positions': [], 'running': False})
        return jsonify(em.get_state())
    
    @app.route('/api/exitmon/settings', methods=['GET', 'POST'])
    def api_exitmon_settings():
        """Get or update Exit Monitor settings (weights, thresholds, etc)."""
        from detection.exit_monitor import get_exit_monitor
        em = get_exit_monitor()
        if not em:
            return jsonify({'ok': False, 'error': 'Not running'})
        if request.method == 'GET':
            return jsonify(em.get_settings())
        data = request.get_json() or {}
        ok = em.update_settings(data)
        return jsonify({'ok': ok, 'settings': em.get_settings()})
    
    @app.route('/api/exitmon/history')
    def api_exitmon_history():
        """Historical Exit Scores for charts."""
        from detection.exit_monitor import get_exit_monitor
        em = get_exit_monitor()
        if not em:
            return jsonify({'history': []})
        date = request.args.get('date', '')
        return jsonify(em.get_history(date))
    
    # ========================================
    # Webhook + Signal Validator Routes
    # ========================================
    
    @app.route('/webhook', methods=['POST'])
    def webhook():
        """Receive TradingView JSON → validate → store result."""
        try:
            import json as _json
            raw = request.get_data(as_text=True)
            data = _json.loads(raw) if raw else {}
            
            print(f"[WEBHOOK] 📥 {data.get('symbol','')} {data.get('action','')}")
            
            from detection.signal_validator import get_validator
            v = get_validator()
            if not v:
                return jsonify({'status': 'error', 'reason': 'Validator not initialized'}), 500
            
            result = v.validate(data)
            return jsonify(result)
        except Exception as e:
            print(f"[WEBHOOK] ❌ Error: {e}")
            return jsonify({'status': 'error', 'reason': str(e)}), 400
    
    @app.route('/api/validator/log')
    def api_validator_log():
        """Signal validation history."""
        from detection.signal_validator import get_validator
        v = get_validator()
        if not v:
            return jsonify({'log': []})
        return jsonify({'log': v.get_log()})
    
    @app.route('/api/validator/delete/<entry_id>', methods=['POST', 'DELETE'])
    def api_validator_delete(entry_id):
        """Delete a single entry."""
        from detection.signal_validator import get_validator
        v = get_validator()
        if not v:
            return jsonify({'ok': False})
        return jsonify({'ok': v.delete_entry(entry_id)})
    
    @app.route('/api/validator/clear', methods=['POST', 'DELETE'])
    def api_validator_clear():
        """Clear all entries."""
        from detection.signal_validator import get_validator
        v = get_validator()
        if not v:
            return jsonify({'ok': False})
        return jsonify({'ok': v.clear_all()})
    
    # ========================================
    # Volume Flow Routes
    # ========================================
    
    @app.route('/api/volume-flow/summary')
    def api_volume_flow_summary():
        from detection.volume_flow import get_volume_flow
        vf = get_volume_flow()
        if not vf:
            return jsonify({'running': False, 'has_data': False})
        return jsonify(vf.get_summary())
    
    @app.route('/api/volume-flow/history')
    def api_volume_flow_history():
        from detection.volume_flow import get_volume_flow
        vf = get_volume_flow()
        if not vf:
            return jsonify({'data': [], 'available_days': []})
        date = request.args.get('date', '')
        return jsonify(vf.get_history(date=date))
    
    # ========================================
    # Funding Rate Monitor Routes
    # ========================================
    
    @app.route('/api/funding/watchlist')
    def api_funding_watchlist():
        """Tracked coins with funding rate history."""
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return jsonify({'running': False, 'coins': [], 'total_tracked': 0})
        return jsonify(fm.get_watchlist())
    
    @app.route('/api/funding/remove/<symbol>', methods=['POST', 'DELETE'])
    def api_funding_remove(symbol):
        """Manually remove coin from watchlist."""
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return jsonify({'ok': False, 'error': 'Not running'})
        if not symbol.endswith('USDT'):
            symbol = symbol.upper() + 'USDT'
        else:
            symbol = symbol.upper()
        ok = fm.remove_coin(symbol)
        return jsonify({'ok': ok, 'symbol': symbol})
    
    @app.route('/api/funding/add/<symbol>', methods=['POST'])
    def api_funding_add(symbol):
        """Manually add coin to watchlist."""
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return jsonify({'ok': False, 'reason': 'Not running'})
        return jsonify(fm.add_coin(symbol))
    
    @app.route('/api/funding/toggle', methods=['GET', 'POST'])
    def api_funding_toggle():
        """Get or set Funding Scanner enabled state."""
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return jsonify({'ok': False, 'enabled': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'enabled': fm.is_enabled(), 'running': fm._running})
        data = request.get_json() or {}
        enabled = bool(data.get('enabled', True))
        fm.set_enabled(enabled)
        return jsonify({'ok': True, 'enabled': fm.is_enabled(), 'running': fm._running})
    
    @app.route('/api/liquidity/toggle', methods=['GET', 'POST'])
    def api_liquidity_toggle():
        """Get or set Liquidity Map enabled state."""
        from detection.liquidity_map import get_liquidity_map
        lm = get_liquidity_map()
        if not lm:
            return jsonify({'ok': False, 'enabled': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'enabled': lm.is_enabled(), 'running': lm._running})
        data = request.get_json() or {}
        enabled = bool(data.get('enabled', True))
        lm.set_enabled(enabled)
        return jsonify({'ok': True, 'enabled': lm.is_enabled(), 'running': lm._running})
    
    @app.route('/api/volume-flow/toggle', methods=['GET', 'POST'])
    def api_volume_flow_toggle():
        """Get or set Volume Flow enabled state."""
        from detection.volume_flow import get_volume_flow
        vf = get_volume_flow()
        if not vf:
            return jsonify({'ok': False, 'enabled': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'enabled': vf.is_enabled(), 'running': vf._running})
        data = request.get_json() or {}
        enabled = bool(data.get('enabled', True))
        vf.set_enabled(enabled)
        return jsonify({'ok': True, 'enabled': vf.is_enabled(), 'running': vf._running})
    
    # ========================================
    # Smart Money Routes
    # ========================================
    
    @app.route('/smart-money')
    def smart_money_page():
        return render_template('smart_money.html')
    
    @app.route('/api/smc/state')
    def api_smc_state():
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'running': False, 'error': 'Not initialized'})
        return jsonify(s.get_state())
    
    @app.route('/api/smc/dedup/reset', methods=['POST'])
    def api_smc_dedup_reset():
        """Reset dedup state for one symbol or all.
        
        Body: {"symbol": "BTCUSDT"}  → reset just that symbol
              {}                      → reset all symbols
        
        After reset, the next signal of any direction will fire (state
        becomes "no prior signal"). Useful when the user knows the trend
        changed but no opposite signal has come through to flip dedup,
        or when re-entering a trade they manually closed.
        """
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'error': 'Scanner not initialised'}), 500
        body = request.get_json(silent=True) or {}
        symbol = (body.get('symbol') or '').strip().upper()
        try:
            # Use the scanner's reset_dedup which PERSISTS the change so it
            # survives restarts and settings reloads (previously this only
            # mutated in-memory state and reverted on the next _load_all_signals).
            res = s.reset_dedup(symbol if symbol else None)
            msg = (f"Dedup reset for {symbol}" if symbol
                   else "Dedup reset for all symbols")
            return jsonify({'ok': True, 'message': msg, 'result': res})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/smc/watchlist', methods=['GET'])
    def api_smc_watchlist():
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'watchlist': []})
        return jsonify({'watchlist': s.get_watchlist()})
    
    @app.route('/api/smc/watchlist/add', methods=['POST'])
    def api_smc_watchlist_add():
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        sym = data.get('symbol', '')
        return jsonify(s.add_symbol(sym))
    
    @app.route('/api/smc/watchlist/remove', methods=['POST'])
    def api_smc_watchlist_remove():
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        sym = data.get('symbol', '')
        return jsonify(s.remove_symbol(sym))
    
    @app.route('/api/smc/settings', methods=['GET', 'POST'])
    def api_smc_settings():
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'settings': s.get_settings()})
        data = request.get_json() or {}
        new_settings = s.update_settings(data)
        return jsonify({'ok': True, 'settings': new_settings})
    
    @app.route('/api/smc/chart')
    def api_smc_chart():
        """Return klines + structure for a symbol. ?symbol=BTCUSDT"""
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'error': 'Not initialized', 'ohlc': []})
        symbol = request.args.get('symbol', 'BTCUSDT')
        return jsonify(s.get_chart_data(symbol))
    
    @app.route('/api/smc/signals/clear', methods=['POST'])
    def api_smc_signals_clear():
        """Clear persisted signal markers. POST {} for all, {'symbol': X} for one."""
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol')
        return jsonify(s.clear_signals(symbol))
    
    @app.route('/api/smc/watchlist/tradeable', methods=['POST'])
    def api_smc_watchlist_tradeable():
        """Toggle tradeable flag for a watchlist symbol.
        Body: {symbol: 'BTCUSDT', tradeable: true|false}
        """
        from detection.smc_scanner import get_smc_scanner
        s = get_smc_scanner()
        if not s:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol', '')
        tradeable = bool(data.get('tradeable', False))
        return jsonify(s.set_tradeable(symbol, tradeable))
    
    # ========================================
    # TOP-100 4H OB Radar — Variant B routes
    # ========================================
    # All endpoints under /api/top100-ob/. Lives as a sub-section of the
    # Smart Money page (no separate top-level page) — reuses smart_money.html.
    
    @app.route('/api/top100-ob/state')
    def api_top100_ob_state():
        """Return scanner settings + last scan summary."""
        from detection.top100_ob_scanner import get_top100_ob_scanner
        try:
            from alerts.telegram_notifier import get_notifier
            scanner = get_top100_ob_scanner(telegram_notifier=get_notifier())
        except Exception:
            scanner = get_top100_ob_scanner()
        return jsonify(scanner.get_settings())
    
    @app.route('/api/top100-ob/settings', methods=['POST'])
    def api_top100_ob_settings():
        """Update scanner settings.
        Body may contain any subset of:
          enabled, telegram_alerts, include_bos_alerts,
          min_quote_volume_usd, top_n,
          timeframe, scan_interval_min, fresh_window_hours,
          long_max_pct, short_min_pct
        Toggling 'enabled' true→false stops the scheduler; false→true starts it.
        Changing 'timeframe' clears stored snapshots (TF-specific data).
        Persists to DB so settings survive worker restart on Render.
        """
        from detection.top100_ob_scanner import get_top100_ob_scanner
        try:
            from alerts.telegram_notifier import get_notifier
            scanner = get_top100_ob_scanner(telegram_notifier=get_notifier())
        except Exception:
            scanner = get_top100_ob_scanner()
        data = request.get_json() or {}
        was_enabled = scanner.get_settings()['enabled']
        new_settings = scanner.update_settings(
            enabled=data.get('enabled'),
            telegram_alerts=data.get('telegram_alerts'),
            include_bos_alerts=data.get('include_bos_alerts'),
            min_quote_volume_usd=data.get('min_quote_volume_usd'),
            top_n=data.get('top_n'),
            timeframe=data.get('timeframe'),
            scan_interval_min=data.get('scan_interval_min'),
            fresh_window_hours=data.get('fresh_window_hours'),
            long_max_pct=data.get('long_max_pct'),
            short_min_pct=data.get('short_min_pct'),
        )
        # Persist to DB. We only write the fields the caller actually
        # tried to change — sending an empty body keeps prior values.
        # Setting names mirror the scanner's internal attrs.
        try:
            db = get_db()
            if 'enabled' in data:
                db.set_setting('top100_ob_enabled',
                               '1' if new_settings['enabled'] else '0')
            if 'telegram_alerts' in data:
                db.set_setting('top100_ob_telegram',
                               '1' if new_settings['telegram_alerts'] else '0')
            if 'include_bos_alerts' in data:
                db.set_setting('top100_ob_include_bos',
                               '1' if new_settings['include_bos_alerts'] else '0')
            if 'min_quote_volume_usd' in data:
                db.set_setting('top100_ob_min_vol_usd',
                               str(int(new_settings['min_quote_volume_usd'])))
            # New settings persistence
            if 'timeframe' in data:
                db.set_setting('top100_ob_timeframe',
                               str(new_settings['timeframe']))
            if 'scan_interval_min' in data:
                db.set_setting('top100_ob_scan_interval_min',
                               str(int(new_settings['scan_interval_min'])))
            if 'fresh_window_hours' in data:
                db.set_setting('top100_ob_fresh_window_hours',
                               str(float(new_settings['fresh_window_hours'])))
            if 'long_max_pct' in data:
                db.set_setting('top100_ob_long_max_pct',
                               str(float(new_settings['long_max_pct'])))
            if 'short_min_pct' in data:
                db.set_setting('top100_ob_short_min_pct',
                               str(float(new_settings['short_min_pct'])))
        except Exception as e:
            print(f"[APP] top100_ob settings persistence error: {e}")
            # Don't fail the request — settings still applied in-memory,
            # they just won't survive a worker restart. User will see
            # them work normally for the rest of this process.
        # Start/stop scheduler when enabled flips
        if not was_enabled and new_settings['enabled']:
            scanner.start()
        elif was_enabled and not new_settings['enabled']:
            scanner.stop()
        return jsonify({'ok': True, 'settings': new_settings})
    
    @app.route('/api/top100-ob/scan', methods=['POST'])
    def api_top100_ob_scan_now():
        """Manual scan trigger ("Refresh now" button). Returns immediately
        with a status — actual scan happens in this thread (might take ~60s).
        Caller should set a long timeout client-side or use the polling
        approach: start scan async (TODO?), then poll /api/top100-ob/state
        for is_scanning=False.
        
        For now we run synchronously — Flask will hold the connection ~60s
        which is fine for a manual user-clicked button. Gunicorn timeout
        on Render is typically 30s default; overridden in Procfile/render.yaml
        if needed. If user reports issues we can switch to async.
        """
        from detection.top100_ob_scanner import get_top100_ob_scanner
        try:
            from alerts.telegram_notifier import get_notifier
            scanner = get_top100_ob_scanner(telegram_notifier=get_notifier())
        except Exception:
            scanner = get_top100_ob_scanner()
        # Run scan in a background thread so we can return immediately and
        # let the UI poll for completion. Otherwise Gunicorn worker would
        # be tied up for 60+ seconds.
        import threading
        def _bg():
            try:
                scanner.scan(triggered_by='manual')
            except Exception as e:
                print(f'[TOP100-OB] Manual scan thread error: {e}')
        t = threading.Thread(target=_bg, name='Top100OB-manual',
                             daemon=True)
        t.start()
        return jsonify({'ok': True, 'message': 'Scan started in background',
                        'is_scanning': True})
    
    @app.route('/api/top100-ob/clear', methods=['POST'])
    def api_top100_ob_clear():
        """Clear all stored Top-100 OB snapshots from the DB.
        
        Use case: user wants a clean slate (e.g., after settings tweak or
        when stale OBs are showing up that no longer reflect current
        market structure). The next scan repopulates from scratch.
        
        We DON'T touch the history table (sob_top100_ob_history) — it's
        an audit log and stays intact for analytics/debugging.
        
        Returns count of rows cleared so the UI can display feedback.
        """
        try:
            db = get_db()
            cleared = db.clear_top100_ob_snapshots()
            print(f'[APP] Top-100 OB snapshots cleared by user '
                  f'({cleared} rows)')
            return jsonify({'ok': True, 'cleared': cleared})
        except Exception as e:
            print(f'[APP] Top-100 OB clear error: {e}')
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    # ====================================================================
    # === Bybit API Credentials management ===
    # ====================================================================
    # Five endpoints power the UI Credentials panel:
    #   GET  /api/credentials/status            — current source + masked preview
    #   POST /api/credentials/save              — plain → encrypt → DB
    #   POST /api/credentials/save-encrypted    — already-encrypted blobs → DB
    #   POST /api/credentials/clear             — remove from DB (fall back to ENV)
    #   POST /api/credentials/test              — call Bybit private API with auth
    #   POST /api/credentials/generate-master   — generate new ENCRYPTION_KEY
    # All endpoints return JSON and never echo secrets back to the client.
    
    def _mask(secret: str) -> str:
        """Return masked preview: '••••••AB12' (8 dots + last 4 chars).
        Empty string returns 'none' so the UI shows a clear 'not set' state.
        """
        if not secret:
            return 'none'
        if len(secret) <= 4:
            return '•' * len(secret)
        return '•' * 8 + secret[-4:]
    
    @app.route('/api/credentials/status')
    def api_credentials_status():
        """Status of Bybit credentials: which source won, masked preview,
        whether ENCRYPTION_KEY is set, whether DB has stored values.
        
        Used by the UI to render the status badge and decide which
        action buttons to enable (e.g., 'Save (encrypted)' is disabled
        when ENCRYPTION_KEY is missing).
        """
        from config.bot_settings import get_bybit_keys, _resolve_bybit_keys_from_db
        from storage.db_operations import get_db
        
        api_key, api_secret = get_bybit_keys()
        
        # Determine source by re-running the resolver paths
        source = 'none'
        if api_key and api_secret:
            db_result = _resolve_bybit_keys_from_db()
            if db_result and db_result[0] == api_key:
                source = 'db_encrypted'
            elif os.environ.get('BYBIT_API_KEY', '').strip() == api_key:
                source = 'env_plain'
            else:
                source = 'env_encrypted'
        
        encryption_key_set = bool(os.environ.get('ENCRYPTION_KEY', '').strip())
        
        # Check if DB has stored encrypted values (regardless of whether
        # they decrypt — user may want to know they're there)
        db_has_stored = False
        try:
            db = get_db()
            db_has_stored = bool(
                db.get_setting('bybit_api_key_encrypted', '')
                and db.get_setting('bybit_api_secret_encrypted', '')
            )
        except Exception:
            pass
        
        return jsonify({
            'ok': True,
            'source': source,                    # 'db_encrypted' | 'env_plain' | 'env_encrypted' | 'none'
            'has_keys': bool(api_key and api_secret),
            'key_preview': _mask(api_key),
            'secret_preview': _mask(api_secret),
            'encryption_key_set': encryption_key_set,
            'db_has_stored': db_has_stored,
            'env_plain_set': bool(os.environ.get('BYBIT_API_KEY', '').strip()
                                  and os.environ.get('BYBIT_API_SECRET', '').strip()),
            'env_encrypted_set': bool(os.environ.get('BYBIT_API_KEY_ENCRYPTED', '').strip()
                                      and os.environ.get('BYBIT_API_SECRET_ENCRYPTED', '').strip()),
        })
    
    @app.route('/api/credentials/save', methods=['POST'])
    def api_credentials_save():
        """Encrypt plain key+secret and save to DB.
        
        Request body (JSON):
          {"api_key": "...", "api_secret": "..."}
        
        Server-side:
          1. Validates ENCRYPTION_KEY is set
          2. Encrypts each with Fernet
          3. Stores encrypted blobs in DB under keys
             `bybit_api_key_encrypted` and `bybit_api_secret_encrypted`
          4. Reloads the Bybit connector with new keys
          5. Returns success + new masked preview
        
        We never echo the plaintext back. The encrypted form is also
        not echoed by default (use save-encrypted endpoint if you need
        to see it). This prevents accidental key exposure in browser
        history / network tabs of shared screens.
        """
        from config.bot_settings import _encrypt_fernet, reload_bybit_keys, get_bybit_keys
        from storage.db_operations import get_db
        from core.bybit_connector import get_connector
        
        enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
        if not enc_key:
            return jsonify({'ok': False,
                'error': 'ENCRYPTION_KEY env var not set. Generate one via '
                         '/api/credentials/generate-master and add it to '
                         'Render env vars first.'}), 400
        
        data = request.get_json(silent=True) or {}
        api_key = (data.get('api_key') or '').strip()
        api_secret = (data.get('api_secret') or '').strip()
        if not api_key or not api_secret:
            return jsonify({'ok': False,
                'error': 'Both api_key and api_secret required'}), 400
        
        # Sanity: Bybit keys are typically 18 chars for key, 36 for secret.
        # Don't enforce strictly (Bybit may change formats), but warn if way off.
        if len(api_key) < 8 or len(api_secret) < 16:
            return jsonify({'ok': False,
                'error': f'Keys look too short (key={len(api_key)} chars, '
                         f'secret={len(api_secret)} chars). Did you paste '
                         f'them correctly?'}), 400
        
        # Encrypt with Fernet
        key_enc = _encrypt_fernet(api_key, enc_key)
        secret_enc = _encrypt_fernet(api_secret, enc_key)
        if not (key_enc and secret_enc):
            return jsonify({'ok': False,
                'error': 'Encryption failed — check ENCRYPTION_KEY format '
                         '(must be valid Fernet base64)'}), 500
        
        # Persist
        try:
            db = get_db()
            db.set_setting('bybit_api_key_encrypted', key_enc)
            db.set_setting('bybit_api_secret_encrypted', secret_enc)
        except Exception as e:
            return jsonify({'ok': False,
                'error': f'DB write failed: {e}'}), 500
        
        # Refresh resolver and Bybit connector
        reload_bybit_keys()
        try:
            connector = get_connector()
            if connector:
                connector.reload_keys()
        except Exception as e:
            # DB save succeeded, connector reload failed — surface but
            # don't fail the whole request. Next restart will pick it up.
            print(f"[CREDS] Connector reload after save failed: {e}")
        
        new_key, _ = get_bybit_keys()
        print(f"[CREDS] ✅ Bybit keys updated via UI (now {len(new_key)} chars, "
              f"source=DB encrypted)")
        return jsonify({
            'ok': True,
            'source': 'db_encrypted',
            'key_preview': _mask(new_key),
            'message': 'Keys encrypted and saved. Active immediately.',
        })
    
    @app.route('/api/credentials/save-encrypted', methods=['POST'])
    def api_credentials_save_encrypted():
        """Save already-encrypted blobs to DB without re-encrypting.
        
        For users who already have Fernet-encrypted values (e.g. from
        a previous deployment's ENV vars they want to migrate to DB).
        
        Request body (JSON):
          {"api_key_encrypted": "gAAAA...", "api_secret_encrypted": "gAAAA..."}
        
        Server verifies they decrypt with the current ENCRYPTION_KEY
        before storing — protects against putting unreachable garbage
        into the DB.
        """
        from config.bot_settings import _decrypt_fernet, reload_bybit_keys
        from storage.db_operations import get_db
        from core.bybit_connector import get_connector
        
        enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
        if not enc_key:
            return jsonify({'ok': False,
                'error': 'ENCRYPTION_KEY env var not set'}), 400
        
        data = request.get_json(silent=True) or {}
        key_enc = (data.get('api_key_encrypted') or '').strip()
        secret_enc = (data.get('api_secret_encrypted') or '').strip()
        if not key_enc or not secret_enc:
            return jsonify({'ok': False,
                'error': 'Both api_key_encrypted and api_secret_encrypted required'}), 400
        
        # Verify both decrypt cleanly — never store unreachable blobs
        test_key = _decrypt_fernet(key_enc, enc_key)
        test_secret = _decrypt_fernet(secret_enc, enc_key)
        if not test_key or not test_secret:
            return jsonify({'ok': False,
                'error': 'Decrypt verification failed — these blobs were '
                         'NOT encrypted with the current ENCRYPTION_KEY'}), 400
        
        try:
            db = get_db()
            db.set_setting('bybit_api_key_encrypted', key_enc)
            db.set_setting('bybit_api_secret_encrypted', secret_enc)
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        
        reload_bybit_keys()
        try:
            c = get_connector()
            if c: c.reload_keys()
        except Exception:
            pass
        print(f"[CREDS] ✅ Encrypted Bybit keys imported to DB ({len(test_key)} chars)")
        return jsonify({
            'ok': True,
            'source': 'db_encrypted',
            'key_preview': _mask(test_key),
        })
    
    @app.route('/api/credentials/clear', methods=['POST'])
    def api_credentials_clear():
        """Delete DB-stored keys, fall back to ENV vars.
        
        Use case: user wants to revert to env-var-based config without
        manually editing the DB. After clear, the next resolver call
        skips the DB tier and uses ENV plain or ENV encrypted.
        """
        from config.bot_settings import reload_bybit_keys
        from storage.db_operations import get_db
        from core.bybit_connector import get_connector
        
        try:
            db = get_db()
            db.set_setting('bybit_api_key_encrypted', '')
            db.set_setting('bybit_api_secret_encrypted', '')
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        
        reload_bybit_keys()
        try:
            c = get_connector()
            if c: c.reload_keys()
        except Exception:
            pass
        print(f"[CREDS] ✅ DB-stored Bybit keys cleared; falling back to ENV")
        return jsonify({'ok': True, 'message': 'DB keys cleared. Falling back to ENV.'})
    
    @app.route('/api/credentials/test', methods=['POST'])
    def api_credentials_test():
        """Test the currently-active Bybit keys by calling a private endpoint.
        
        We use `get_wallet_balance` which requires auth but doesn't move
        any money. Errors are sanitized — we return Bybit's `retCode`
        and `retMsg` plus a friendly mapped explanation (e.g.,
        33004 → 'API key expired').
        """
        from core.bybit_connector import get_connector
        c = get_connector()
        if c is None:
            return jsonify({'ok': False, 'error': 'Connector not initialised'}), 500
        if not (c.api_key and c.api_secret):
            return jsonify({'ok': False, 'error': 'No keys configured'}), 400
        try:
            # Try a lightweight private call. Use whatever pybit exposes for
            # account info — wallet balance is universal.
            r = c.session.get_wallet_balance(accountType='UNIFIED')
            ret_code = r.get('retCode', -1)
            ret_msg = r.get('retMsg', 'unknown')
            if ret_code == 0:
                # Auth works — but don't echo any balance numbers back.
                return jsonify({
                    'ok': True,
                    'auth_works': True,
                    'message': 'Authentication successful — keys are valid',
                    'key_preview': _mask(c.api_key),
                })
            else:
                # Common error codes:
                #   10003 — invalid API key
                #   33004 — API key expired
                #   10004 — sign error (wrong secret)
                #   10005 — permissions error
                #   10006 — rate limited
                friendly = {
                    10003: 'Invalid API key',
                    10004: 'Sign error (wrong API secret)',
                    10005: 'Permission denied — key missing Read+Trade scope',
                    10006: 'Rate limited — try again in a few seconds',
                    33004: 'API key EXPIRED — create new ones on Bybit',
                }.get(ret_code, ret_msg)
                return jsonify({
                    'ok': False,
                    'auth_works': False,
                    'ret_code': ret_code,
                    'ret_msg': ret_msg,
                    'friendly': friendly,
                })
        except Exception as e:
            err = str(e)
            return jsonify({
                'ok': False,
                'auth_works': False,
                'error': err[:300],  # avoid huge stack traces in UI
            }), 200
    
    # ========================================================================
    # CoinGecko API key (single key) — status / save / clear
    # ========================================================================
    @app.route('/api/credentials/coingecko/status')
    def api_cg_cred_status():
        from config.bot_settings import get_coingecko_key, _resolve_coingecko_key_from_db
        from storage.db_operations import get_db
        key = get_coingecko_key()
        source = 'none'
        if key:
            if _resolve_coingecko_key_from_db() == key:
                source = 'db_encrypted'
            elif os.environ.get('COINGECKO_API_KEY', '').strip() == key:
                source = 'env_plain'
        db_has = False
        try:
            db = get_db()
            db_has = bool(db.get_setting('coingecko_api_key_encrypted', ''))
        except Exception:
            pass
        return jsonify({
            'ok': True,
            'source': source,
            'has_key': bool(key),
            'key_preview': _mask(key),
            'encryption_key_set': bool(os.environ.get('ENCRYPTION_KEY', '').strip()),
            'db_has_stored': db_has,
            'env_plain_set': bool(os.environ.get('COINGECKO_API_KEY', '').strip()),
        })

    @app.route('/api/credentials/coingecko/save', methods=['POST'])
    def api_cg_cred_save():
        """Encrypt + store CoinGecko API key in DB. Body: {api_key}."""
        from config.bot_settings import _encrypt_fernet
        from storage.db_operations import get_db
        enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
        if not enc_key:
            return jsonify({'ok': False, 'error': 'ENCRYPTION_KEY env var not set'}), 400
        data = request.get_json(silent=True) or {}
        api_key = (data.get('api_key') or '').strip()
        if not api_key:
            return jsonify({'ok': False, 'error': 'api_key required'}), 400
        enc = _encrypt_fernet(api_key, enc_key)
        if not enc:
            return jsonify({'ok': False, 'error': 'Encryption failed'}), 500
        try:
            get_db().set_setting('coingecko_api_key_encrypted', enc)
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        # Hot-reload the CoinGecko client so the new key applies immediately
        try:
            from detection.coingecko_client import get_coingecko_client
            get_coingecko_client().reload_key()
        except Exception:
            pass
        print('[CREDS] ✅ CoinGecko key encrypted + stored in DB')
        return jsonify({'ok': True, 'key_preview': _mask(api_key)})

    @app.route('/api/credentials/coingecko/clear', methods=['POST'])
    def api_cg_cred_clear():
        from storage.db_operations import get_db
        try:
            get_db().set_setting('coingecko_api_key_encrypted', '')
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        try:
            from detection.coingecko_client import get_coingecko_client
            get_coingecko_client().reload_key()
        except Exception:
            pass
        print('[CREDS] ✅ CoinGecko DB key cleared; falling back to ENV')
        return jsonify({'ok': True, 'message': 'CoinGecko DB key cleared.'})

    # ========================================================================
    # Binance API keys (key + secret) — status / save / clear
    # NOTE: bot uses Binance public endpoints only; keys give higher rate
    # limits but aren't functionally required.
    # ========================================================================
    @app.route('/api/credentials/binance/status')
    def api_bn_cred_status():
        from config.bot_settings import get_binance_keys, _resolve_binance_keys_from_db
        from storage.db_operations import get_db
        api_key, api_secret = get_binance_keys()
        source = 'none'
        if api_key and api_secret:
            db_result = _resolve_binance_keys_from_db()
            if db_result and db_result[0] == api_key:
                source = 'db_encrypted'
            elif os.environ.get('BINANCE_API_KEY', '').strip() == api_key:
                source = 'env_plain'
        db_has = False
        try:
            db = get_db()
            db_has = bool(db.get_setting('binance_api_key_encrypted', '')
                          and db.get_setting('binance_api_secret_encrypted', ''))
        except Exception:
            pass
        return jsonify({
            'ok': True,
            'source': source,
            'has_keys': bool(api_key and api_secret),
            'key_preview': _mask(api_key),
            'secret_preview': _mask(api_secret),
            'encryption_key_set': bool(os.environ.get('ENCRYPTION_KEY', '').strip()),
            'db_has_stored': db_has,
            'env_plain_set': bool(os.environ.get('BINANCE_API_KEY', '').strip()
                                  and os.environ.get('BINANCE_API_SECRET', '').strip()),
        })

    @app.route('/api/credentials/binance/save', methods=['POST'])
    def api_bn_cred_save():
        """Encrypt + store Binance key+secret in DB. Body: {api_key, api_secret}."""
        from config.bot_settings import _encrypt_fernet
        from storage.db_operations import get_db
        enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
        if not enc_key:
            return jsonify({'ok': False, 'error': 'ENCRYPTION_KEY env var not set'}), 400
        data = request.get_json(silent=True) or {}
        api_key = (data.get('api_key') or '').strip()
        api_secret = (data.get('api_secret') or '').strip()
        if not api_key or not api_secret:
            return jsonify({'ok': False, 'error': 'Both api_key and api_secret required'}), 400
        k_enc = _encrypt_fernet(api_key, enc_key)
        s_enc = _encrypt_fernet(api_secret, enc_key)
        if not (k_enc and s_enc):
            return jsonify({'ok': False, 'error': 'Encryption failed'}), 500
        try:
            db = get_db()
            db.set_setting('binance_api_key_encrypted', k_enc)
            db.set_setting('binance_api_secret_encrypted', s_enc)
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        print('[CREDS] ✅ Binance keys encrypted + stored in DB '
              '(takes effect on next restart for the connector)')
        return jsonify({'ok': True, 'key_preview': _mask(api_key),
                        'note': 'Saved. Restart or redeploy to apply to the '
                                'Binance connector (public API works regardless).'})

    @app.route('/api/credentials/binance/clear', methods=['POST'])
    def api_bn_cred_clear():
        from storage.db_operations import get_db
        try:
            db = get_db()
            db.set_setting('binance_api_key_encrypted', '')
            db.set_setting('binance_api_secret_encrypted', '')
        except Exception as e:
            return jsonify({'ok': False, 'error': f'DB write failed: {e}'}), 500
        print('[CREDS] ✅ Binance DB keys cleared; falling back to ENV')
        return jsonify({'ok': True, 'message': 'Binance DB keys cleared.'})

    @app.route('/api/credentials/generate-master', methods=['POST'])
    def api_credentials_generate_master():
        """Generate a new Fernet master key for ENCRYPTION_KEY env var.
        
        Returns the generated key to display in the UI. User copies it
        to Render env vars manually (we can't write env vars from inside
        the app — that's a Render dashboard action). After they save
        the env var and redeploy, the new ENCRYPTION_KEY is active.
        
        Note: rotating the master key requires re-encrypting all stored
        keys. To keep this simple, we ALSO return a flag indicating
        whether the user has stored DB keys (which would become
        unreadable after a key rotation).
        """
        try:
            from cryptography.fernet import Fernet
            new_key = Fernet.generate_key().decode()
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
        
        # Check whether user has DB-encrypted keys (which would need re-encryption)
        try:
            from storage.db_operations import get_db
            db = get_db()
            has_db_keys = bool(
                db.get_setting('bybit_api_key_encrypted', '')
                and db.get_setting('bybit_api_secret_encrypted', '')
            )
        except Exception:
            has_db_keys = False
        
        return jsonify({
            'ok': True,
            'encryption_key': new_key,
            'has_stored_keys': has_db_keys,
            'warning': ('You have DB-stored encrypted keys. Changing ENCRYPTION_KEY '
                        'will make them unreadable. Re-save your plain keys after '
                        'updating the env var.') if has_db_keys else None,
        })
    
    @app.route('/api/bybit/perpetual-symbols')
    def api_bybit_perpetual_symbols():
        """Return the set of USDT-perpetual symbols available on Bybit
        Futures, used by the Top-100 OB radar to filter out coins that
        only exist on Binance (which the radar scans against).
        
        Response shape:
          { "symbols": ["BTCUSDT", "ETHUSDT", ...], "count": N }
        
        Cached server-side for 5 minutes (in BybitConnector). Frontend
        caches the result in-memory for the session.
        """
        try:
            from core.bybit_connector import get_connector
            bc = get_connector()
            symbols = bc.get_perpetual_symbols()
            return jsonify({
                'symbols': sorted(list(symbols)),
                'count': len(symbols),
            })
        except Exception as e:
            print(f'[APP] Bybit perpetual symbols fetch error: {e}')
            # Return empty list — frontend treats this as "filter
            # disabled" (everything passes through), which is safer
            # than blanking the table on a transient API hiccup.
            return jsonify({'symbols': [], 'count': 0,
                            'error': str(e)}), 200
    
    @app.route('/api/top100-ob/snapshots')
    def api_top100_ob_snapshots():
        """List current snapshots. Query params:
          only_with_ob (bool): default true — hide rows with no OB
          min_quote_volume (float): override the configured min
        """
        from detection.top100_ob_scanner import get_top100_ob_scanner
        from storage.db_operations import get_db
        scanner = get_top100_ob_scanner()
        # Default to filtering to symbols with active OBs — that's the
        # interesting view. UI can flip the toggle to see all.
        only_with_ob_q = request.args.get('only_with_ob', 'true').lower()
        only_with_ob = (only_with_ob_q != 'false')
        try:
            min_vol = float(request.args.get('min_quote_volume',
                                             scanner._min_quote_volume_usd))
        except (TypeError, ValueError):
            min_vol = scanner._min_quote_volume_usd
        rows = get_db().list_top100_ob_snapshots(
            only_with_ob=only_with_ob, min_quote_volume=min_vol)
        return jsonify({'ok': True, 'count': len(rows), 'snapshots': rows})
    
    @app.route('/api/top100-ob/history')
    def api_top100_ob_history():
        """Recent OB lifecycle events (the "Recent Discoveries" feed).
        Query params:
          hours (int): lookback window, default 24
          event_types (csv): filter to specific events, e.g. 'created,replaced'
          limit (int): max rows, default 100
        """
        from storage.db_operations import get_db
        try:
            hours = int(request.args.get('hours', '24'))
        except (TypeError, ValueError):
            hours = 24
        types_arg = request.args.get('event_types', '')
        types = [t.strip() for t in types_arg.split(',') if t.strip()] or None
        try:
            limit = min(500, int(request.args.get('limit', '100')))
        except (TypeError, ValueError):
            limit = 100
        rows = get_db().list_top100_ob_history(
            hours=hours, event_types=types, limit=limit)
        return jsonify({'ok': True, 'count': len(rows), 'history': rows})
    
    @app.route('/api/top100-ob/add-to-watchlist', methods=['POST'])
    def api_top100_ob_add_to_watchlist():
        """Push a symbol from the TOP-100 view into the SMC scanner
        watchlist. Convenience for "Add to SMC Watchlist" button.
        Body: {symbol: 'BTCUSDT'}
        """
        from detection.smc_scanner import get_smc_scanner
        smc = get_smc_scanner()
        if not smc:
            return jsonify({'ok': False, 'reason': 'SMC scanner not initialized'})
        data = request.get_json() or {}
        symbol = (data.get('symbol') or '').upper().strip()
        if not symbol:
            return jsonify({'ok': False, 'reason': 'Missing symbol'})
        return jsonify(smc.add_symbol(symbol))
    
    # ===== Trade Manager =====
    
    @app.route('/api/tm/state')
    def api_tm_state():
        """Return Trade Manager state: positions, closed trades, stats."""
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        return jsonify(tm.get_state())
    
    @app.route('/api/tm/settings', methods=['GET', 'POST'])
    def api_tm_settings():
        """Get or update Trade Manager settings."""
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'settings': tm.get_settings()})
        new_settings = request.get_json() or {}
        return jsonify({'ok': True, 'settings': tm.update_settings(new_settings)})
    
    @app.route('/api/tm/positions/open', methods=['POST'])
    def api_tm_open_position():
        """User-initiated manual position open from the Decision Center
        panel. Uses Position Sizing + SL/TP settings from TM exactly the
        same way an auto-opened (signal-driven) position would.
        
        Body: { symbol: 'BTCUSDT', side: 'LONG' | 'SHORT' }
        Returns: { ok: bool, reason?: str, position?: {...}, entry_price?: float }
        
        Failure cases (returned as ok=False with reason):
          - TM not enabled
          - Bybit not configured (no API key)
          - Position already exists for this symbol (real or shadow)
          - Bad inputs / order placement failed
        """
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'TM not initialized'})
        data = request.get_json() or {}
        return jsonify(tm.manual_open(
            symbol=data.get('symbol', ''),
            side=data.get('side', ''),
        ))
    
    @app.route('/api/tm/positions/close', methods=['POST'])
    def api_tm_position_close():
        """Manually close an open position. Body: {symbol: 'BTCUSDT'}"""
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'ok': False, 'reason': 'symbol required'})
        return jsonify(tm.manual_close(symbol))
    
    @app.route('/api/tm/shadow/close', methods=['POST'])
    def api_tm_shadow_close():
        """Manually close an open paper-trading position. Body: {symbol: 'BTCUSDT'}"""
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'ok': False, 'reason': 'symbol required'})
        return jsonify(tm.manual_close_shadow(symbol))
    
    @app.route('/api/tm/positions/manual-sl-tp', methods=['POST'])
    def api_tm_manual_sl_tp():
        """Set or clear per-position manual SL/TP override.
        
        Body: {
            symbol: 'BTCUSDT',
            manual_sl?: <float>,    # absolute price, 0/'' clears
            manual_tp?: <float>,    # absolute price, 0/'' clears
            is_shadow?: bool        # default False (real position)
        }
        
        Either or both of manual_sl/manual_tp can be in the body.
        Omitting a field leaves it unchanged on the position; passing
        0 or empty string explicitly clears it.
        """
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'ok': False, 'reason': 'symbol required'})
        return jsonify(tm.update_manual_sl_tp(
            symbol=symbol,
            manual_sl=data.get('manual_sl'),
            manual_tp=data.get('manual_tp'),
            is_shadow=bool(data.get('is_shadow')),
        ))
    
    @app.route('/api/tm/positions/manual-mode', methods=['POST'])
    def api_tm_manual_mode():
        """Toggle per-position manual mode.
        
        Body: {
            symbol: 'BTCUSDT',
            enabled: bool,          # True = manual mode ON, False = OFF
            is_shadow?: bool        # default False (real position)
        }
        
        When manual mode is ON for a position:
          - Automatic exits (SL, TP, time stop, HTF flip, Reverse SMC,
            CHoCH/BOS exits, trailing, BE) are bypassed.
          - Only Manual SL / Manual TP price levels OR force-close via UI
            can close the position.
          - New SMC signals on this symbol are ignored (no auto-reverse).
        
        When OFF (default), the full automatic strategy logic runs.
        """
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'ok': False, 'reason': 'symbol required'})
        return jsonify(tm.update_manual_mode(
            symbol=symbol,
            enabled=bool(data.get('enabled')),
            is_shadow=bool(data.get('is_shadow')),
        ))
    
    @app.route('/api/tm/closed/delete', methods=['POST'])
    def api_tm_closed_delete():
        """Permanently delete a real-trade entry from Recent Closed Trades.
        Body: {index: <int>}. Stats are recomputed automatically.
        """
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        idx = data.get('index')
        if idx is None:
            return jsonify({'ok': False, 'reason': 'index required'})
        return jsonify(tm.delete_closed_trade(idx))
    
    @app.route('/api/tm/shadow_closed/delete', methods=['POST'])
    def api_tm_shadow_closed_delete():
        """Permanently delete a paper-trade entry from Recent Paper Closes.
        Body: {index: <int>}. Stats are recomputed automatically.
        """
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Not initialized'})
        data = request.get_json() or {}
        idx = data.get('index')
        if idx is None:
            return jsonify({'ok': False, 'reason': 'index required'})
        return jsonify(tm.delete_shadow_closed_trade(idx))
    
    @app.route('/api/validator/toggle', methods=['GET', 'POST'])
    def api_validator_toggle():
        """Get or set Signal Validator (TradingView Signals) enabled state."""
        from detection.signal_validator import get_validator
        v = get_validator()
        if not v:
            return jsonify({'ok': False, 'enabled': False, 'reason': 'Not initialized'})
        if request.method == 'GET':
            return jsonify({'ok': True, 'enabled': v.is_enabled()})
        data = request.get_json() or {}
        enabled = bool(data.get('enabled', True))
        v.set_enabled(enabled)
        return jsonify({'ok': True, 'enabled': v.is_enabled()})
    
    @app.route('/api/funding/coin/<symbol>')
    def api_funding_coin(symbol):
        """Full rate history for a single coin (for chart)."""
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return jsonify({'found': False})
        if not symbol.endswith('USDT'):
            symbol = symbol.upper() + 'USDT'
        else:
            symbol = symbol.upper()
        return jsonify(fm.get_coin_rates(symbol))
    
    @app.route('/api/funding/flow/<symbol>')
    def api_funding_flow(symbol):
        """Volume flow analysis for a watchlist coin."""
        from detection.coin_flow import get_coin_flow
        cf = get_coin_flow()
        if not cf:
            return jsonify({'found': False})
        if not symbol.endswith('USDT'):
            symbol = symbol.upper() + 'USDT'
        else:
            symbol = symbol.upper()
        return jsonify(cf.get_coin_summary(symbol))
    
    @app.route('/api/funding/signals')
    def api_funding_signals():
        """All coin flow signals summary."""
        from detection.coin_flow import get_coin_flow
        cf = get_coin_flow()
        if not cf:
            return jsonify({'signals': {}, 'running': False})
        return jsonify(cf.get_all_signals())
    
    # ========================================
    # QM Zone Hunter Routes
    # ========================================
    
    @app.route('/qm')
    def qm_page():
        """QM Zone Hunter page"""
        from scheduler.qm_job import get_qm_job, QM_SETTINGS_DEFAULTS
        import json
        
        db = get_db()
        
        # Watchlist
        watchlist_str = db.get_setting('qm_watchlist', '')
        if not watchlist_str:
            watchlist_str = db.get_setting('ctr_watchlist', '')
        watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
        
        # Settings (з дефолтами)
        settings = {}
        for key, default in QM_SETTINGS_DEFAULTS.items():
            settings[key] = db.get_setting(key, default)
        
        # Scan results
        scan_results_str = db.get_setting('qm_last_scan', '[]')
        try:
            scan_results = json.loads(scan_results_str)
        except:
            scan_results = []
        
        # Last scan time
        last_scan_time = db.get_setting('qm_last_scan_time', None)
        if last_scan_time:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(last_scan_time.replace('Z', '+00:00'))
                last_scan_time = dt.strftime('%H:%M:%S')
            except:
                pass
        
        # Status
        try:
            qm_job = get_qm_job(db)
            qm_running = qm_job.is_running()
            status = qm_job.get_status()
            stats = status.get('stats', {})
        except:
            qm_running = False
            stats = {}
        
        # Signals
        signals_str = db.get_setting('qm_signals', '[]')
        try:
            qm_signals = json.loads(signals_str)
            qm_signals = sorted(qm_signals, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
        except:
            qm_signals = []
        
        return render_template('qm.html',
            watchlist=watchlist,
            scan_results=scan_results,
            last_scan_time=last_scan_time,
            qm_running=qm_running,
            zones_active=stats.get('zones_active', 0),
            signals_sent=stats.get('signals_sent', 0),
            settings=settings,
            qm_signals=qm_signals,
        )
    
    @app.route('/api/qm/start', methods=['POST'])
    def api_qm_start():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        try:
            job = get_qm_job(db)
            ok = job.start()
            return jsonify({'ok': ok})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/stop', methods=['POST'])
    def api_qm_stop():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        try:
            job = get_qm_job(db)
            job.stop()
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/scan', methods=['POST'])
    def api_qm_scan():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        try:
            job = get_qm_job(db)
            results = job.scan_now()
            return jsonify({'ok': True, 'signals': len(results)})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/results')
    def api_qm_results():
        from scheduler.qm_job import get_qm_job
        import json
        db = get_db()
        try:
            job = get_qm_job(db)
            status = job.get_status()
            stats = status.get('stats', {})
            scan_time = db.get_setting('qm_last_scan_time', '')
            if scan_time:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
                    scan_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            # Per-symbol scan results
            scan_results = job.get_results() if job.is_running() else []
            
            return jsonify({
                'scan_time': scan_time,
                'zones_active': stats.get('zones_active', 0),
                'signals_sent': stats.get('signals_sent', 0),
                'scans': stats.get('scans', 0),
                'patterns_found': stats.get('patterns_found', 0),
                'last_scan_ms': stats.get('last_scan_ms', 0),
                'scan_results': scan_results,
            })
        except:
            return jsonify({})
    
    @app.route('/api/qm/watchlist/add', methods=['POST'])
    def api_qm_watchlist_add():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        if not symbol:
            return jsonify({'ok': False, 'error': 'Empty symbol'})
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        try:
            job = get_qm_job(db)
            job.add_symbol(symbol)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/watchlist/remove', methods=['POST'])
    def api_qm_watchlist_remove():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        try:
            job = get_qm_job(db)
            job.remove_symbol(symbol)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/settings', methods=['POST'])
    def api_qm_settings():
        from scheduler.qm_job import get_qm_job, QM_SETTINGS_DEFAULTS
        db = get_db()
        data = request.get_json()
        
        saved = 0
        for key in QM_SETTINGS_DEFAULTS:
            if key in data:
                db.set_setting(key, str(data[key]))
                saved += 1
        
        try:
            job = get_qm_job(db)
            job.reload_settings()
        except:
            pass
        
        return jsonify({'ok': True, 'saved': saved})
    
    @app.route('/api/qm/signals/delete', methods=['POST'])
    def api_qm_signals_delete():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        data = request.get_json()
        timestamp = data.get('timestamp')
        if not timestamp:
            return jsonify({'ok': False})
        try:
            job = get_qm_job(db)
            ok = job.delete_signal(timestamp)
            return jsonify({'ok': ok})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    @app.route('/api/qm/signals/clear', methods=['POST'])
    def api_qm_signals_clear():
        from scheduler.qm_job import get_qm_job
        db = get_db()
        try:
            job = get_qm_job(db)
            count = job.clear_signals()
            return jsonify({'ok': True, 'cleared': count})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    
    # ============================================================
    # Volumized OB Radar — 6 endpoints (mirrors TOP-100 OB Radar style)
    # ============================================================
    
    @app.route('/api/vol-radar/state')
    def api_volradar_state():
        """One-shot dashboard payload: settings + active items + recent
        snapshots + top stats. UI polls every ~30s."""
        try:
            from detection.volumized_ob_radar import get_volumized_ob_radar
            radar = get_volumized_ob_radar()
            return jsonify({'ok': True, **radar.get_state()})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/vol-radar/settings', methods=['POST'])
    def api_volradar_settings():
        """Update settings. Body: any subset of settings keys (see
        radar.get_settings() for the full key list)."""
        try:
            from detection.volumized_ob_radar import get_volumized_ob_radar
            radar = get_volumized_ob_radar()
            data = request.get_json() or {}
            new_settings = radar.update_settings(**data)
            return jsonify({'ok': True, 'settings': new_settings})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/vol-radar/copy-from-main', methods=['POST'])
    def api_volradar_copy_from_main():
        """Pull Volumized algorithm params from the main SMC scanner's
        settings — single-button convenience that mirrors swing_length,
        ob_end_method, max_atr_mult, zone_count, combine_obs, timeframe."""
        try:
            from detection.volumized_ob_radar import get_volumized_ob_radar
            radar = get_volumized_ob_radar()
            new_settings = radar.copy_from_main_volumized()
            return jsonify({'ok': True, 'settings': new_settings})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/vol-radar/scan', methods=['POST'])
    def api_volradar_scan():
        """Trigger a manual scan tick. Blocks until done; large scans (~100
        symbols × 600ms throttle) can take 60+ seconds. Frontend should
        show a spinner. Returns the same summary shape as scheduled scans."""
        try:
            from detection.volumized_ob_radar import get_volumized_ob_radar
            radar = get_volumized_ob_radar()
            summary = radar.scan(triggered_by='manual')
            return jsonify({'ok': True, 'summary': summary})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/vol-radar/remove', methods=['POST'])
    def api_volradar_remove():
        """Manual remove from radar tracking. Body: {symbol: 'BTCUSDT'}.
        Removes from watchlist + bumps manual counter + starts cooldown.
        Same effect as user clicking 🗑 in the watchlist UI."""
        try:
            from detection.volumized_ob_radar import get_volumized_ob_radar
            data = request.get_json() or {}
            symbol = data.get('symbol', '')
            if not symbol:
                return jsonify({'ok': False, 'reason': 'symbol required'})
            radar = get_volumized_ob_radar()
            return jsonify(radar.manual_remove(symbol))
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/vol-radar/stats/<symbol>')
    def api_volradar_stat(symbol):
        """Per-symbol lifetime stats. Useful for "how often does radar
        pick BTC, and what's its conversion rate?" — shown in tooltip."""
        try:
            from storage.db_operations import get_db
            stat = get_db().volradar_get_stat(symbol.upper())
            if stat is None:
                return jsonify({'ok': True, 'stat': None})
            return jsonify({'ok': True, 'stat': stat})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    # ========================================================================
    # Liquidation Map endpoints
    # ========================================================================
    
    @app.route('/api/liquidation-map/state')
    def api_liqmap_state():
        """Return full liquidation map state for the requested symbol.
        Query params:
          symbol (default BTCUSDT)
          lookback (hours; default 24; allowed 1, 4, 24, 168)
          include_mitigated (default 0; pass 1 to also see hit-through buckets)
        """
        try:
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm is None:
                return jsonify({'ok': False, 'reason': 'Daemon not initialized'})
            
            symbol = request.args.get('symbol', 'BTCUSDT').upper()
            try:
                lookback = int(request.args.get('lookback', 24))
            except ValueError:
                lookback = 24
            include_mitigated = request.args.get('include_mitigated', '0') == '1'
            
            # Auto-register the symbol as on-demand if it's not already
            # being tracked — so first /state call seeds it.
            from detection.liquidation_map.liquidation_map import BACKGROUND_SYMBOLS
            if symbol not in BACKGROUND_SYMBOLS:
                lm.request_symbol(symbol)
            
            state = lm.get_state(symbol=symbol,
                                   lookback_hours=lookback,
                                   include_mitigated=include_mitigated)
            return jsonify({'ok': True, **state})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liquidation-map/request-symbol', methods=['POST'])
    def api_liqmap_request_symbol():
        """UI calls this when user types a new symbol into the input. Adds
        (or refreshes TTL of) the symbol on the daemon's active list."""
        try:
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm is None:
                return jsonify({'ok': False, 'reason': 'Daemon not initialized'})
            data = request.get_json() or {}
            symbol = data.get('symbol', '')
            if not symbol:
                return jsonify({'ok': False, 'reason': 'symbol required'})
            return jsonify(lm.request_symbol(symbol))
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liquidation-map/active-symbols')
    def api_liqmap_active_symbols():
        """List of symbols currently being scanned. Helpful for the UI to
        decide whether to wait for "building history" or render immediately."""
        try:
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm is None:
                return jsonify({'ok': False, 'reason': 'Daemon not initialized',
                                 'symbols': []})
            return jsonify({'ok': True, 'symbols': lm.get_active_symbols()})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e), 'symbols': []})
    
    @app.route('/api/liquidation-map/price-series')
    def api_liqmap_price_series():
        """Recent price series for the chart's price-action overlay. Returns
        OHLC candles for the requested interval, sorted ascending by ts.

        Query params:
          symbol:   default BTCUSDT
          hours:    lookback window in hours (default 24)
          interval: 1m | 5m | 15m | 30m | 1h | 4h (default 15m)

        Response shape:
          {ok, interval, series: [{ts, o, h, l, c}]}

        The 'close' field is also included for backward-compatibility with
        old chart code that may still expect a thin line; new candle renderer
        uses o/h/l/c.
        """
        try:
            symbol = request.args.get('symbol', 'BTCUSDT').upper()
            try:
                hours = int(request.args.get('hours', 24))
            except ValueError:
                hours = 24
            interval = request.args.get('interval', '15m')
            if interval not in ('1m', '5m', '15m', '30m', '1h', '4h'):
                interval = '15m'

            # Cap kline count per interval to keep response small while
            # covering the chosen window
            bars_per_hour = {'1m': 60, '5m': 12, '15m': 4, '30m': 2,
                              '1h': 1, '4h': 0.25}
            limit = int(min(1000, max(20, hours * bars_per_hour[interval])))

            from detection.market_data import get_market_data
            md = get_market_data()
            if md is None:
                return jsonify({'ok': False, 'interval': interval, 'series': []})
            klines = md.fetch_klines(symbol, interval=interval, limit=limit)
            if not klines:
                return jsonify({'ok': True, 'interval': interval, 'series': []})
            series = []
            for k in klines:
                ts = k.get('t') or k.get('ts') or k.get('open_time')
                if ts and ts > 1e12:  # ms → s
                    ts = int(ts / 1000)
                o = k.get('o') or k.get('open')
                h = k.get('h') or k.get('high')
                l = k.get('l') or k.get('low')
                # CRITICAL: market_data.fetch_klines returns bars with the
                # close price under key 'p' (see its docstring: bars are
                # {p, v, b, s, h, l, o, t}). The previous mapping only
                # checked 'c'/'close', so EVERY bar was skipped and the
                # endpoint always returned an empty series — that's why
                # the dashboard chart never showed any candles.
                c = k.get('c') or k.get('close') or k.get('p')
                if ts and c:
                    try:
                        series.append({
                            'ts': int(ts),
                            'o': float(o if o is not None else c),
                            'h': float(h if h is not None else c),
                            'l': float(l if l is not None else c),
                            'c': float(c),
                            # legacy alias
                            'close': float(c),
                        })
                    except (TypeError, ValueError):
                        continue
            return jsonify({'ok': True, 'interval': interval, 'series': series})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e), 'series': []})
    
    @app.route('/api/orderbook/walls')
    def api_orderbook_walls():
        """Live order-book walls for a single symbol.
        
        PRIMARY source (2026-06-09+): Bybit REST orderbook, 500 levels/side.
        Depth20 WS gave only ±0.01% coverage on BTC — every wall collapsed
        onto the mid-price stripe. Bybit deep book spans ±0.5–15% depending
        on symbol tick density, which makes walls actually spread across
        the chart like the user's reference.
        
        FALLBACK: Binance WS depth20 collector (kept for resilience if
        Bybit REST ever fails).
        
        Query params:
          symbol  (required, default BTCUSDT)
          top_n   (optional, default 8 — strongest walls per side)
        
        Response:
          {ok, symbol, walls: {bid_walls, ask_walls, mid_price, imbalance_pct, ...},
           source: 'bybit_rest' | 'binance_ws'}
        """
        try:
            symbol = request.args.get('symbol', 'BTCUSDT').upper().strip()
            if not symbol or len(symbol) > 20:
                return jsonify({'ok': False, 'reason': 'Invalid symbol'})
            try:
                top_n = int(request.args.get('top_n', 8))
            except ValueError:
                top_n = 8
            top_n = max(1, min(top_n, 20))
            
            from detection.orderbook_collector import (
                get_orderbook_collector, fetch_bybit_orderbook,
                fetch_aggregated_orderbook, compute_walls_buckets,
                compute_walls_buckets_v3)
            obc = get_orderbook_collector()
            
            # === Primary: aggregated Bybit+OKX+Hyperliquid deep book ===
            # Walls get per-exchange USD attribution; a wall standing on
            # 2-3 venues simultaneously is a stronger signal than a
            # single-venue one. Bucket params identical to v2.
            snapshot = fetch_aggregated_orderbook(symbol)
            if snapshot is not None:
                walls = compute_walls_buckets_v3(snapshot, top_n=top_n)
                if walls:
                    # Feed the spoof tracker on every poll and ship its
                    # rolling stats in the same response (no extra request).
                    try:
                        from detection.manipulation_tracker import get_manipulation_tracker
                        mt = get_manipulation_tracker()
                        mt.update(symbol, walls)
                        walls['manipulation'] = mt.get_state(symbol)
                    except Exception as _me:
                        print(f"[OBC] manipulation tracker error: {_me}")
                    return jsonify({
                        'ok': True,
                        'symbol': symbol,
                        'walls': walls,
                        'pending': False,
                        'source': 'aggregated:' + '+'.join(walls.get('sources') or []),
                    })
            
            # === Secondary: Bybit alone (if parallel fetch all failed
            # but a cached Bybit snapshot can still be had) ===
            snapshot = fetch_bybit_orderbook(symbol)
            if snapshot is not None:
                walls = compute_walls_buckets(snapshot, top_n=top_n)
                if walls:
                    return jsonify({
                        'ok': True,
                        'symbol': symbol,
                        'walls': walls,
                        'pending': False,
                        'source': 'bybit_rest',
                    })
            
            # === Fallback: Binance WS depth20 ===
            snapshot = obc.request(symbol)
            if snapshot is None:
                return jsonify({
                    'ok': True,
                    'symbol': symbol,
                    'walls': None,
                    'pending': True,
                    'hint': 'Bybit REST unavailable; subscribed to Binance WS, awaiting snapshot',
                })
            walls = obc.compute_walls(snapshot, top_n=top_n)
            return jsonify({
                'ok': True,
                'symbol': symbol,
                'walls': walls,
                'pending': False,
                'source': 'binance_ws',
            })
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e), 'walls': None})
    
    @app.route('/api/orderbook/status')
    def api_orderbook_status():
        """Debug endpoint — which symbols are currently subscribed and
        whether their WS is healthy. Useful for spotting stuck connections."""
        try:
            from detection.orderbook_collector import get_orderbook_collector
            obc = get_orderbook_collector()
            return jsonify({
                'ok': True,
                'active': obc.active_symbols(),
            })
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/health/apis')
    def api_health_apis():
        """Current status of every external API the bot depends on."""
        from detection.api_health import get_api_health_monitor
        mon = get_api_health_monitor()
        if mon is None:
            return jsonify({'running': False, 'services': []})
        return jsonify(mon.get_status())
    
    @app.route('/api/liqmap-signal/status')
    def api_liqsig_status():
        """Scanner status + latest scores per watchlist symbol — feeds the
        watchlist buttons (score badges) and the alerts toggle state."""
        try:
            from detection.liqmap_signal_scanner import get_liqmap_signal_scanner
            sc = get_liqmap_signal_scanner()
            if sc is None:
                return jsonify({'ok': False, 'reason': 'not initialized'})
            return jsonify({'ok': True, **sc.get_status()})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liqmap-signal/settings', methods=['POST'])
    def api_liqsig_settings():
        """Update alert settings: {enabled?, threshold?, cooldown_min?}"""
        try:
            data = request.get_json() or {}
            db = get_db()
            if 'enabled' in data:
                db.set_setting('liqmap_signal_enabled',
                                'true' if data['enabled'] else 'false')
            if 'threshold' in data:
                t = max(50, min(95, int(data['threshold'])))
                db.set_setting('liqmap_signal_threshold', str(t))
            if 'cooldown_min' in data:
                c = max(5, min(720, int(data['cooldown_min'])))
                db.set_setting('liqmap_signal_cooldown', str(c))
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liqmap-signal/watchlist')
    def api_liqsig_watchlist():
        """SMC watchlist symbols — feeds the quick-switch buttons row."""
        try:
            import json as _json
            raw = get_db().get_setting('smc_watchlist', '[]')
            wl = _json.loads(raw) if isinstance(raw, str) else (raw or [])
            return jsonify({'ok': True,
                             'symbols': [str(s).upper() for s in wl if s]})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e), 'symbols': []})
    
    # Squeeze cache: (symbol, interval) -> {'ts': float, 'result': dict}.
    # Klines fetch costs one Bybit call; 60s TTL matches how fast a 15m
    # squeeze can meaningfully change.
    _squeeze_cache = {}
    
    @app.route('/api/squeeze')
    def api_squeeze():
        """TTM Squeeze (BB-inside-Keltner compression) for one symbol+TF.
        Feeds the SQUEEZE gauge on the dashboard Liquidity Map.
        
        Query: symbol (default BTCUSDT), interval (default 15m)
        Response: {ok, symbol, interval, squeeze_on, probability,
                   bars_in_squeeze, momentum_rising, momentum_side}
        """
        try:
            import time as _sqt
            symbol = request.args.get('symbol', 'BTCUSDT').upper().strip()
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            interval = request.args.get('interval', '15m')
            if interval not in ('1m', '5m', '15m', '30m', '1h', '4h'):
                interval = '15m'
            
            key = (symbol, interval)
            now = _sqt.time()
            cached = _squeeze_cache.get(key)
            if cached and (now - cached['ts']) < 60:
                return jsonify(cached['result'])
            
            from detection.market_data import get_market_data
            from detection.squeeze import calc_squeeze
            md = get_market_data()
            if md is None:
                return jsonify({'ok': False, 'reason': 'market data unavailable'})
            klines = md.fetch_klines(symbol, interval=interval, limit=120)
            # Multi-band TTM (see squeeze.py): graded compression bands
            # make higher TFs readable — 'low/mid/high' instead of the
            # old binary 1.5-cross that never fired on 1h/4h.
            sq = calc_squeeze(klines or [])
            
            # === Directional Squeeze Probability (2026-06-12) ===
            # Calibrated to the reference semantics ("squeeze probability"
            # = how likely price gets PULLED INTO the nearer liquidation
            # fuel), built from data we already have:
            #   fuel_dir  — decayed liq-levels above vs below mark, each
            #               level weighted by 1/(1+dist%/2): a $2M pool
            #               1% away pulls harder than $20M at 12%;
            #   compression — TTM probability (the timing factor);
            #   momentum  — agreement bonus when TTM momentum points the
            #               same way as the fuel.
            # prob = 100×clamp(0.55·|fuel_dir| + 0.25·comp + 0.20·agree).
            # No liq data → falls back to compression-only, undirected.
            sq_dir, sq_prob = 'flat', sq['probability']
            fuel_above = fuel_below = 0.0
            try:
                from detection.liquidation_map.liquidation_map import (
                    get_liquidation_map)
                lm = get_liquidation_map()
                if lm is not None:
                    lstate = lm.get_state(symbol, lookback_hours=24)
                    mark = lstate.get('mark_price')
                    for lev in (lstate.get('levels') or []):
                        if not mark:
                            break
                        dist_pct = abs(lev['price'] - mark) / mark * 100.0
                        if dist_pct > 15:
                            continue
                        wgt = lev['usd'] / (1.0 + dist_pct / 2.0)
                        if lev['price'] > mark:
                            fuel_above += wgt
                        else:
                            fuel_below += wgt
                    den = fuel_above + fuel_below
                    if den > 0:
                        fuel_dir = (fuel_above - fuel_below) / den
                        comp = sq['probability'] / 100.0
                        mom = (1 if sq['momentum'] > 0 else
                               -1 if sq['momentum'] < 0 else 0)
                        agree = 1.0 if (mom != 0 and
                                        mom == (1 if fuel_dir > 0 else -1)
                                        ) else 0.0
                        raw = (0.55 * abs(fuel_dir) + 0.25 * comp
                               + 0.20 * agree)
                        sq_prob = round(max(0.0, min(1.0, raw)) * 100.0, 1)
                        sq_dir = ('long' if fuel_dir > 0.1 else
                                  'short' if fuel_dir < -0.1 else 'flat')
            except Exception:
                pass
            result = {
                'ok': sq['ok'],
                'symbol': symbol,
                'interval': interval,
                'squeeze_on': sq['squeeze_on'],
                'band': sq.get('band', 'off'),
                'probability': sq['probability'],
                # Directional probability (reference-calibrated). Old
                # fields above untouched — the signal scanner keeps using
                # 'probability' (compression energy) unchanged.
                'squeeze_prob': sq_prob,
                'squeeze_dir': sq_dir,
                'fuel_above_usd': round(fuel_above, 0),
                'fuel_below_usd': round(fuel_below, 0),
                'bars_in_squeeze': sq['bars_in_squeeze'],
                'momentum_rising': sq['momentum_rising'],
                'momentum_side': 1 if sq['momentum'] > 0 else (-1 if sq['momentum'] < 0 else 0),
                'reason': sq.get('reason', ''),
            }
            _squeeze_cache[key] = {'ts': now, 'result': result}
            # Bound cache
            if len(_squeeze_cache) > 100:
                oldest = min(_squeeze_cache, key=lambda k: _squeeze_cache[k]['ts'])
                _squeeze_cache.pop(oldest, None)
            return jsonify(result)
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    # Throttle map for on-demand forecast computes: symbol -> last compute ts.
    # The dashboard polls every 2s; a fresh compute fetches 1H+4H klines, so
    # we only allow one per symbol per 120s. Watchlist symbols are already
    # refreshed by the SMC scanner on its own cadence and hit the cache path.
    _forecast_ondemand_ts = {}
    
    @app.route('/api/forecast')
    def api_forecast():
        """Forecast 1H/4H for one symbol — feeds the AI Market Pressure bar
        on the dashboard Liquidity Map.
        
        Cache-first (SMC scanner keeps watchlist symbols fresh). For symbols
        outside the watchlist, computes on-demand at most once per 120s.
        
        Response: {ok, symbol, forecast_1h: {side,pct,confidence},
                   forecast_4h: {...}, computed_at}
        """
        try:
            symbol = request.args.get('symbol', 'BTCUSDT').upper().strip()
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe is None:
                return jsonify({'ok': False, 'reason': 'ForecastEngine not initialized'})
            
            import time as _ftime
            cached = fe.get(symbol)
            now = _ftime.time()
            # Refresh if: nothing cached, or cache older than 10 min — but
            # never more often than once per 120s per symbol from this path.
            stale = (cached is None
                     or (now - cached.get('computed_at', 0)) > 600)
            if stale:
                last = _forecast_ondemand_ts.get(symbol, 0)
                if now - last >= 120:
                    _forecast_ondemand_ts[symbol] = now
                    try:
                        cached = fe.update(symbol)
                    except Exception as e:
                        print(f"[Forecast API] on-demand compute {symbol} failed: {e}")
            
            if not cached:
                return jsonify({'ok': True, 'symbol': symbol,
                                 'forecast_1h': None, 'forecast_4h': None,
                                 'pending': True})
            return jsonify({
                'ok': True,
                'symbol': symbol,
                'forecast_1h': cached.get('forecast_1h'),
                'forecast_4h': cached.get('forecast_4h'),
                'computed_at': cached.get('computed_at'),
            })
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liquidation-map/status')
    def api_liqmap_status():
        """Authoritative state for the UI:
          - enabled_setting: what the persisted DB setting says
          - daemon_running: whether the singleton's thread is alive
          - oi_age_sec: how long since the last OI snapshot (any provider)
          - events_count: how many active events exist for the symbol
        UI uses this on init + on toggle to show the right overlay state."""
        import time as _t
        try:
            symbol = request.args.get('symbol', 'BTCUSDT').upper()
            from storage.db_operations import get_db
            db = get_db()
            enabled = db.get_setting('liquidation_map_enabled', '1') == '1'
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            daemon_running = bool(lm and lm.is_running())
            # Latest OI snapshot age (across all providers — pick min)
            oi_age = None
            if lm:
                for p in lm.providers:
                    row = db.liqmap_get_latest_oi(symbol, p.name)
                    if row and row.get('ts'):
                        age = int(_t.time()) - row['ts']
                        if oi_age is None or age < oi_age:
                            oi_age = age
            # Count active events for symbol in last 24h
            ev = db.liqmap_get_events(
                symbol, lookback_seconds=86400,
                include_mitigated=False, limit=10000)
            events_count = len(ev)
            return jsonify({
                'ok': True,
                'symbol': symbol,
                'enabled_setting': enabled,
                'daemon_running': daemon_running,
                'oi_age_sec': oi_age,
                'events_count': events_count,
                'scan_interval_sec': 60,
            })
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liquidation-map/poc')
    def api_liqmap_poc():
        """Volume Profile: POC (Point of Control), VAH/VAL (Value Area).
        
        Standard market-profile algorithm:
          1. Pull klines for the lookback window with an interval scaled
             to keep response size reasonable (1m for short, 1h for long).
          2. For each kline, distribute its volume evenly across the
             price buckets spanned by [low, high]. This is the textbook
             "TPO-equivalent" volume profile.
          3. POC = bucket with max accumulated volume.
          4. Value Area = expand outward from POC (always picking the side
             with higher adjacent volume) until 70% of total volume captured.
             VAH and VAL are the high/low of that expanded range.
        
        Returns:
          {ok, poc, vah, val, total_volume, value_area_pct, bucket_size}
        """
        try:
            symbol = request.args.get('symbol', 'BTCUSDT').upper()
            try:
                hours = int(request.args.get('hours', 24))
            except ValueError:
                hours = 24
            
            # Scale interval to keep response size sane across timeframes
            if hours <= 4:
                interval, limit = '1m', min(hours * 60, 240)
            elif hours <= 24:
                interval, limit = '5m', min(hours * 12, 300)
            else:
                interval, limit = '1h', min(hours, 200)
            
            from detection.market_data import get_market_data
            md = get_market_data()
            if md is None:
                return jsonify({'ok': False, 'reason': 'market_data unavailable'})
            klines = md.fetch_klines(symbol, interval=interval, limit=limit)
            if not klines:
                return jsonify({'ok': False, 'reason': 'no klines'})
            
            # Bucket size — matches the liquidation_map convention so POC
            # lines visually align with cluster bands on the same chart
            sym_upper = symbol.upper()
            if sym_upper.startswith('BTC'):
                bucket_size = 25.0
            elif sym_upper.startswith('ETH'):
                bucket_size = 1.0
            else:
                # Use mid price of first kline as anchor for 0.03% sizing
                ref_price = float(klines[0].get('c') or klines[0].get('close', 0))
                bucket_size = max(round(ref_price * 0.0003, 4), 0.0001)
            
            # Aggregate volume per price-bucket. Spread each kline's volume
            # uniformly across the buckets it touched ([low, high]).
            bucket_vol = {}
            for k in klines:
                high = float(k.get('h') or k.get('high', 0))
                low  = float(k.get('l') or k.get('low', 0))
                vol  = float(k.get('v') or k.get('volume', 0))
                if high <= 0 or low <= 0 or vol <= 0:
                    continue
                if high < low:
                    high, low = low, high
                # Number of buckets the candle range covers (inclusive)
                n = max(1, int((high - low) / bucket_size) + 1)
                per_bucket = vol / n
                for j in range(n):
                    price = low + j * bucket_size
                    # Round to bucket midpoint
                    bk = round(price / bucket_size) * bucket_size + bucket_size / 2
                    bucket_vol[bk] = bucket_vol.get(bk, 0) + per_bucket
            
            if not bucket_vol:
                return jsonify({'ok': False, 'reason': 'empty profile'})
            
            # POC = max-volume bucket
            sorted_prices = sorted(bucket_vol.keys())
            poc_price = max(bucket_vol, key=bucket_vol.get)
            poc_idx = sorted_prices.index(poc_price)
            
            total_vol = sum(bucket_vol.values())
            value_area_target = total_vol * 0.70
            captured = bucket_vol[poc_price]
            lo_idx = poc_idx
            hi_idx = poc_idx
            # Expand outward, always picking the side with higher volume
            while captured < value_area_target:
                upper = bucket_vol[sorted_prices[hi_idx + 1]] if hi_idx + 1 < len(sorted_prices) else -1
                lower = bucket_vol[sorted_prices[lo_idx - 1]] if lo_idx - 1 >= 0 else -1
                if upper < 0 and lower < 0:
                    break
                if upper >= lower:
                    hi_idx += 1
                    captured += upper
                else:
                    lo_idx -= 1
                    captured += lower
            
            return jsonify({
                'ok': True,
                'symbol': symbol,
                'poc': round(poc_price, 4),
                'vah': round(sorted_prices[hi_idx], 4),
                'val': round(sorted_prices[lo_idx], 4),
                'total_volume': round(total_vol, 4),
                'value_area_pct': round(captured / total_vol if total_vol else 0, 4),
                'bucket_size': bucket_size,
                'lookback_hours': hours,
                'interval': interval,
                'kline_count': len(klines),
            })
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
    @app.route('/api/liquidation-map/toggle', methods=['POST'])
    def api_liqmap_toggle():
        """Enable/disable the daemon globally (persisted in DB setting)."""
        try:
            from storage.db_operations import get_db
            data = request.get_json() or {}
            enabled = bool(data.get('enabled'))
            get_db().set_setting('liquidation_map_enabled',
                                   '1' if enabled else '0')
            # Start/stop the daemon to honor the new setting immediately
            from detection.liquidation_map import (
                get_liquidation_map, init_liquidation_map)
            from detection.market_data import get_market_data
            lm = get_liquidation_map()
            if enabled:
                if lm is None:
                    lm = init_liquidation_map(db=get_db(),
                                                market_data=get_market_data())
                if not lm.is_running():
                    lm.start()
            else:
                if lm is not None and lm.is_running():
                    lm.stop()
            return jsonify({'ok': True, 'enabled': enabled})
        except Exception as e:
            return jsonify({'ok': False, 'reason': str(e)})
    
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
