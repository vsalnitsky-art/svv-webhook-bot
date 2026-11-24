"""
CTR API Endpoints - Additional routes for CTR Scanner v8.3.0

Add these routes to flask_app.py after existing CTR routes.
Also update the ctr_page() function to pass zone data and use new signal format.
"""

# ==========================================
# ADD THESE ROUTES AFTER @app.route('/api/ctr/only-mode')
# ==========================================

@app.route('/api/ctr/signals/delete/<int:signal_id>', methods=['POST'])
def api_ctr_signal_delete(signal_id):
    """Delete a specific CTR signal"""
    from scheduler.ctr_job import get_ctr_job
    
    try:
        job = get_ctr_job(db)
        success = job.delete_signal(signal_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ctr/signals/clear', methods=['POST'])
def api_ctr_signals_clear():
    """Clear all CTR signals or for a specific symbol"""
    from scheduler.ctr_job import get_ctr_job
    
    data = request.get_json() or {}
    symbol = data.get('symbol')  # Optional - clear only for this symbol
    
    try:
        job = get_ctr_job(db)
        count = job.clear_signals(symbol)
        return jsonify({'success': True, 'cleared': count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==========================================
# UPDATED ctr_page() FUNCTION
# Replace the existing ctr_page() with this version
# ==========================================

@app.route('/ctr')
def ctr_page():
    """CTR Scanner page with zones and signals management"""
    from scheduler.ctr_job import get_ctr_job
    
    # Get watchlist
    watchlist_str = db.get_setting('ctr_watchlist', '')
    watchlist = [s.strip().upper() for s in watchlist_str.split(',') if s.strip()]
    
    # Get scan results with zone info
    scan_results_str = db.get_setting('ctr_last_scan', '[]')
    try:
        scan_results = json.loads(scan_results_str)
    except:
        scan_results = []
    
    last_scan_time = db.get_setting('ctr_last_scan_time', None)
    
    # Get status
    try:
        ctr_job = get_ctr_job(db)
        ctr_running = ctr_job.is_running()
        ctr_status = ctr_job.get_status()
        ws_connected = ctr_status.get('ws_connected', False)
        stats = ctr_status.get('stats', {})
    except:
        ctr_running = False
        ctr_status = {}
        ws_connected = False
        stats = {}
    
    # CTR Only Mode
    ctr_only_mode = db.get_setting('ctr_only_mode', '0')
    if isinstance(ctr_only_mode, str):
        ctr_only_mode = ctr_only_mode in ('1', 'true', 'yes')
    
    # SMC Filter
    smc_enabled_str = db.get_setting('ctr_smc_filter_enabled', '0')
    smc_filter_enabled = smc_enabled_str in ('1', 'true', 'True', 'yes')
    
    # Settings
    settings = {
        'ctr_timeframe': db.get_setting('ctr_timeframe', '15m'),
        'ctr_fast_length': db.get_setting('ctr_fast_length', 21),
        'ctr_slow_length': db.get_setting('ctr_slow_length', 50),
        'ctr_cycle_length': db.get_setting('ctr_cycle_length', 10),
        'ctr_upper': db.get_setting('ctr_upper', 75),
        'ctr_lower': db.get_setting('ctr_lower', 25),
        # SMC
        'ctr_smc_filter_enabled': smc_filter_enabled,
        'ctr_smc_swing_length': db.get_setting('ctr_smc_swing_length', 50),
        'ctr_smc_zone_threshold': db.get_setting('ctr_smc_zone_threshold', 1.0),
    }
    
    # Get signals - try new DB method first, fallback to settings
    try:
        ctr_signals = db.get_ctr_signals(limit=20)
    except AttributeError:
        # Fallback to old method
        signals_str = db.get_setting('ctr_signals', '[]')
        try:
            ctr_signals = json.loads(signals_str)
            ctr_signals = sorted(ctr_signals, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
        except:
            ctr_signals = []
    
    return render_template('ctr.html',
        watchlist=watchlist,
        scan_results=scan_results,
        last_scan_time=last_scan_time,
        settings=settings,
        ctr_running=ctr_running,
        ws_connected=ws_connected,
        stats=stats,
        ctr_only_mode=ctr_only_mode,
        smc_filter_enabled=smc_filter_enabled,
        ctr_signals=ctr_signals
    )
