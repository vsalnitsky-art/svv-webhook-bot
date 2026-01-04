"""
Diagnostic endpoints for troubleshooting
"""
from flask import Blueprint, jsonify
import traceback

diagnostic_bp = Blueprint('diagnostic', __name__)

@diagnostic_bp.route('/api/diagnostic/bybit')
def test_bybit_api():
    """Test Bybit API connectivity"""
    results = {
        'tickers': {'status': 'pending'},
        'klines': {'status': 'pending'},
        'funding': {'status': 'pending'},
    }
    
    try:
        from core.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        
        # Test 1: Get tickers
        try:
            symbols = fetcher.get_top_symbols(limit=5, min_volume=1000000)
            results['tickers'] = {
                'status': 'ok',
                'count': len(symbols),
                'sample': [s['symbol'] for s in symbols[:3]] if symbols else []
            }
        except Exception as e:
            results['tickers'] = {'status': 'error', 'error': str(e)}
        
        # Test 2: Get klines
        if results['tickers']['status'] == 'ok' and results['tickers']['count'] > 0:
            try:
                sym = results['tickers']['sample'][0]
                klines = fetcher.get_klines(sym, '240', limit=10)
                results['klines'] = {
                    'status': 'ok',
                    'symbol': sym,
                    'count': len(klines),
                    'last_close': klines[-1]['close'] if klines else None
                }
            except Exception as e:
                results['klines'] = {'status': 'error', 'error': str(e)}
        
        # Test 3: Get funding rate
        if results['tickers']['status'] == 'ok' and results['tickers']['count'] > 0:
            try:
                sym = results['tickers']['sample'][0]
                funding = fetcher.get_funding_rate(sym)
                results['funding'] = {
                    'status': 'ok',
                    'symbol': sym,
                    'rate': funding
                }
            except Exception as e:
                results['funding'] = {'status': 'error', 'error': str(e)}
        
        return jsonify({
            'success': True,
            'results': results,
            'all_ok': all(r['status'] == 'ok' for r in results.values())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@diagnostic_bp.route('/api/diagnostic/scan-test')
def test_scan():
    """Run a mini scan test"""
    try:
        from detection.sleeper_scanner import get_sleeper_scanner
        from storage import get_db
        
        db = get_db()
        scanner = get_sleeper_scanner()
        
        # Get just 3 symbols
        symbols = scanner.fetcher.get_top_symbols(limit=3, min_volume=50000000)
        
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No symbols returned from API',
                'api_working': False
            })
        
        results = []
        for sym_data in symbols:
            try:
                result = scanner._analyze_symbol(sym_data)
                results.append({
                    'symbol': sym_data['symbol'],
                    'analyzed': True,
                    'passed': result is not None,
                    'score': result['total_score'] if result else None,
                    'direction': result['direction'] if result else None
                })
            except Exception as e:
                results.append({
                    'symbol': sym_data['symbol'],
                    'analyzed': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'symbols_found': len(symbols),
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@diagnostic_bp.route('/api/diagnostic/scheduler')
def test_scheduler():
    """Check scheduler status in detail"""
    try:
        from scheduler.background_jobs import get_scheduler
        
        scheduler = get_scheduler()
        
        jobs_info = []
        for job in scheduler.scheduler.get_jobs():
            jobs_info.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger),
                'pending': job.pending
            })
        
        return jsonify({
            'success': True,
            'is_running': scheduler.is_running,
            'jobs_count': len(jobs_info),
            'jobs': jobs_info,
            'job_stats': scheduler.job_stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
