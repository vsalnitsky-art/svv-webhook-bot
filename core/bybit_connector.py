"""
Bybit Connector - API client for Bybit exchange
"""
import time
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Any
from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_CONFIG, API_LIMITS

class BybitConnector:
    """Bybit API client with rate limiting and error handling"""
    
    BASE_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.api_key = api_key or BYBIT_API_KEY
        self.api_secret = api_secret or BYBIT_API_SECRET
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.recv_window = BYBIT_CONFIG['recv_window']
        self.timeout = BYBIT_CONFIG['timeout']
        self.session = requests.Session()
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                      signed: bool = False) -> Dict:
        """Make API request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            params['api_key'] = self.api_key
            params['timestamp'] = timestamp
            params['recv_window'] = self.recv_window
            params['sign'] = self._generate_signature(params)
        
        for attempt in range(API_LIMITS['max_retries']):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=params, timeout=self.timeout)
                
                data = response.json()
                
                if data.get('retCode') == 0:
                    return data.get('result', data)
                else:
                    error_msg = data.get('retMsg', 'Unknown error')
                    if attempt < API_LIMITS['max_retries'] - 1:
                        time.sleep(API_LIMITS['rate_limit_delay'] * (attempt + 1))
                        continue
                    raise Exception(f"API Error: {error_msg}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < API_LIMITS['max_retries'] - 1:
                    time.sleep(API_LIMITS['rate_limit_delay'] * (attempt + 1))
                    continue
                raise Exception(f"Request failed: {str(e)}")
        
        return {}
    
    # === PUBLIC ENDPOINTS ===
    
    def get_tickers(self, category: str = "linear") -> List[Dict]:
        """Get all tickers for category"""
        result = self._make_request('GET', '/v5/market/tickers', {'category': category})
        return result.get('list', [])
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                   category: str = "linear") -> List[Dict]:
        """Get kline/candlestick data"""
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, API_LIMITS['kline_limit'])
        }
        result = self._make_request('GET', '/v5/market/kline', params)
        
        # Convert to list of dicts
        klines = []
        for k in result.get('list', []):
            klines.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'turnover': float(k[6]) if len(k) > 6 else 0
            })
        return list(reversed(klines))  # Oldest first
    
    def get_orderbook(self, symbol: str, limit: int = 25, category: str = "linear") -> Dict:
        """Get order book"""
        params = {'category': category, 'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/v5/market/orderbook', params)
    
    def get_instrument_info(self, symbol: str = None, category: str = "linear") -> List[Dict]:
        """Get instrument info"""
        params = {'category': category}
        if symbol:
            params['symbol'] = symbol
        result = self._make_request('GET', '/v5/market/instruments-info', params)
        return result.get('list', [])
    
    def get_funding_rate(self, symbol: str, category: str = "linear") -> Dict:
        """Get current funding rate"""
        params = {'category': category, 'symbol': symbol}
        result = self._make_request('GET', '/v5/market/funding/history', params)
        if result.get('list'):
            return result['list'][0]
        return {}
    
    def get_open_interest(self, symbol: str, interval: str = "1h", 
                          limit: int = 48, category: str = "linear") -> List[Dict]:
        """Get open interest history"""
        params = {
            'category': category,
            'symbol': symbol,
            'intervalTime': interval,
            'limit': limit
        }
        result = self._make_request('GET', '/v5/market/open-interest', params)
        return result.get('list', [])
    
    # === PRIVATE ENDPOINTS (require API keys) ===
    
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        """Get wallet balance"""
        params = {'accountType': account_type}
        result = self._make_request('GET', '/v5/account/wallet-balance', params, signed=True)
        
        if result.get('list'):
            account = result['list'][0]
            return {
                'total_equity': float(account.get('totalEquity', 0)),
                'available_balance': float(account.get('totalAvailableBalance', 0)),
                'total_margin': float(account.get('totalMarginBalance', 0)),
                'unrealized_pnl': float(account.get('totalPerpUPL', 0)),
            }
        return {'total_equity': 0, 'available_balance': 0, 'total_margin': 0, 'unrealized_pnl': 0}
    
    def get_positions(self, symbol: str = None, category: str = "linear") -> List[Dict]:
        """Get open positions"""
        params = {'category': category, 'settleCoin': 'USDT'}
        if symbol:
            params['symbol'] = symbol
        result = self._make_request('GET', '/v5/position/list', params, signed=True)
        
        positions = []
        for p in result.get('list', []):
            if float(p.get('size', 0)) > 0:
                positions.append({
                    'symbol': p['symbol'],
                    'side': p['side'],
                    'size': float(p['size']),
                    'entry_price': float(p.get('avgPrice', 0)),
                    'mark_price': float(p.get('markPrice', 0)),
                    'unrealized_pnl': float(p.get('unrealisedPnl', 0)),
                    'leverage': int(p.get('leverage', 1)),
                    'liq_price': float(p.get('liqPrice', 0)) if p.get('liqPrice') else None,
                })
        return positions
    
    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "Market",
                    price: float = None, stop_loss: float = None, take_profit: float = None,
                    reduce_only: bool = False, category: str = "linear") -> Dict:
        """Place a new order"""
        params = {
            'category': category,
            'symbol': symbol,
            'side': side,  # Buy or Sell
            'orderType': order_type,
            'qty': str(qty),
            'timeInForce': 'GTC',
            'reduceOnly': reduce_only,
        }
        
        if price and order_type == "Limit":
            params['price'] = str(price)
        
        if stop_loss:
            params['stopLoss'] = str(stop_loss)
        if take_profit:
            params['takeProfit'] = str(take_profit)
        
        return self._make_request('POST', '/v5/order/create', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: str, category: str = "linear") -> Dict:
        """Cancel an order"""
        params = {
            'category': category,
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('POST', '/v5/order/cancel', params, signed=True)
    
    def cancel_all_orders(self, symbol: str = None, category: str = "linear") -> Dict:
        """Cancel all orders"""
        params = {'category': category}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('POST', '/v5/order/cancel-all', params, signed=True)
    
    def set_leverage(self, symbol: str, leverage: int, category: str = "linear") -> Dict:
        """Set leverage for a symbol"""
        params = {
            'category': category,
            'symbol': symbol,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }
        return self._make_request('POST', '/v5/position/set-leverage', params, signed=True)
    
    def set_trading_stop(self, symbol: str, stop_loss: float = None, 
                         take_profit: float = None, trailing_stop: float = None,
                         category: str = "linear") -> Dict:
        """Set trading stop for a position"""
        params = {
            'category': category,
            'symbol': symbol,
            'positionIdx': 0
        }
        if stop_loss:
            params['stopLoss'] = str(stop_loss)
        if take_profit:
            params['takeProfit'] = str(take_profit)
        if trailing_stop:
            params['trailingStop'] = str(trailing_stop)
        
        return self._make_request('POST', '/v5/position/trading-stop', params, signed=True)
    
    def close_position(self, symbol: str, side: str, qty: float, category: str = "linear") -> Dict:
        """Close a position"""
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="Market",
            reduce_only=True,
            category=category
        )
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            result = self._make_request('GET', '/v5/market/time')
            return True
        except:
            return False


# Singleton instance
_connector_instance = None

def get_connector(api_key: str = None, api_secret: str = None, testnet: bool = False) -> BybitConnector:
    """Get or create Bybit connector instance"""
    global _connector_instance
    if _connector_instance is None or api_key:
        _connector_instance = BybitConnector(api_key, api_secret, testnet)
    return _connector_instance
