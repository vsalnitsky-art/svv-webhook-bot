"""
Diagnostic endpoints for troubleshooting
"""
from flask import Blueprint, jsonify
import traceback

diagnostic_bp = Blueprint('diagnostic', __name__)

@diagnostic_bp.route('/api/diagnostic/binance')
def test_binance_api():
    """Test Binance Futures API connectivity (для сканування)"""
    results = {
        'tickers': {'status': 'pending'},
        'klines': {'status': 'pending'},
        'funding': {'status': 'pending'},
        'open_interest': {'status': 'pending'},
    }
    
    try:
        from core.binance_connector import get_binance_connector
        connector = get_binance_connector()
        
        # Test 1: Connection test
        if not connector.test_connection():
            return jsonify({
                'success': False,
                'error': 'Binance connection test failed',
                'results': results
            })
        
        # Test 2: Get tickers
        try:
            tickers = connector.get_tickers()
            # Filter USDT pairs with good volume
            usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_tickers.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            top_symbols = usdt_tickers[:5]
            
            results['tickers'] = {
                'status': 'ok',
                'count': len(usdt_tickers),
                'sample': [s['symbol'] for s in top_symbols]
            }
        except Exception as e:
            results['tickers'] = {'status': 'error', 'error': str(e)}
        
        # Test 3: Get klines
        if results['tickers']['status'] == 'ok' and results['tickers']['count'] > 0:
            try:
                sym = results['tickers']['sample'][0]
                klines = connector.get_klines(sym, '240', limit=10)
                results['klines'] = {
                    'status': 'ok',
                    'symbol': sym,
                    'count': len(klines),
                    'last_close': klines[-1]['close'] if klines else None
                }
            except Exception as e:
                results['klines'] = {'status': 'error', 'error': str(e)}
        
        # Test 4: Get funding rate
        if results['tickers']['status'] == 'ok' and results['tickers']['count'] > 0:
            try:
                sym = results['tickers']['sample'][0]
                funding = connector.get_funding_rate(sym)
                results['funding'] = {
                    'status': 'ok',
                    'symbol': sym,
                    'rate': funding.get('funding_rate') if funding else None
                }
            except Exception as e:
                results['funding'] = {'status': 'error', 'error': str(e)}
        
        # Test 5: Get Open Interest
        if results['tickers']['status'] == 'ok' and results['tickers']['count'] > 0:
            try:
                sym = results['tickers']['sample'][0]
                oi = connector.get_current_open_interest(sym)
                results['open_interest'] = {
                    'status': 'ok',
                    'symbol': sym,
                    'oi': oi.get('open_interest') if oi else None
                }
            except Exception as e:
                results['open_interest'] = {'status': 'error', 'error': str(e)}
        
        return jsonify({
            'success': True,
            'exchange': 'Binance Futures',
            'purpose': 'Scanning & Analysis',
            'results': results,
            'all_ok': all(r['status'] == 'ok' for r in results.values())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@diagnostic_bp.route('/api/diagnostic/bybit')
def test_bybit_api():
    """Test Bybit API connectivity (для торгівлі)"""
    results = {
        'tickers': {'status': 'pending'},
        'balance': {'status': 'pending'},
        'positions': {'status': 'pending'},
    }
    
    try:
        from core.bybit_connector import get_connector
        connector = get_connector()
        
        # Test 1: Get tickers
        try:
            tickers = connector.get_tickers()
            usdt_tickers = [t for t in tickers if t.get('symbol', '').endswith('USDT')]
            results['tickers'] = {
                'status': 'ok',
                'count': len(usdt_tickers),
                'sample': [t['symbol'] for t in usdt_tickers[:3]]
            }
        except Exception as e:
            results['tickers'] = {'status': 'error', 'error': str(e)}
        
        # Test 2: Get balance (requires API key)
        try:
            balance = connector.get_wallet_balance()
            results['balance'] = {
                'status': 'ok',
                'balance': balance if balance else 'No API key'
            }
        except Exception as e:
            results['balance'] = {'status': 'error', 'error': str(e)}
        
        # Test 3: Get positions (requires API key)
        try:
            positions = connector.get_positions()
            results['positions'] = {
                'status': 'ok',
                'count': len(positions) if positions else 0
            }
        except Exception as e:
            results['positions'] = {'status': 'error', 'error': str(e)}
        
        return jsonify({
            'success': True,
            'exchange': 'Bybit',
            'purpose': 'Trading',
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
