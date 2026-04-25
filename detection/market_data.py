"""
Market Data Provider v1.1 — Multi-Exchange with Fallback

Primary: Binance Futures (free, no key)
Fallback 1: OKX (free, no key)
Fallback 2: Bybit (free, no key) — for Bybit-exclusive coins

Tracks which exchange provided data for UI display.
"""

import requests
import time
from typing import Dict, List, Optional, Tuple

REQUEST_TIMEOUT = 10
DELAY = 0.3

# Binance endpoints
BN_KLINE   = 'https://fapi.binance.com/fapi/v1/klines'
BN_OI      = 'https://fapi.binance.com/fapi/v1/openInterest'
BN_LS      = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
BN_TOP_LS  = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'
BN_TAKER   = 'https://fapi.binance.com/futures/data/takerlongshortRatio'
BN_DEPTH   = 'https://fapi.binance.com/fapi/v1/depth'

# OKX endpoints
OKX_KLINE  = 'https://www.okx.com/api/v5/market/candles'
OKX_OI     = 'https://www.okx.com/api/v5/rubik/stat/contracts-open-interest-history'
OKX_LS     = 'https://www.okx.com/api/v5/rubik/stat/contracts-long-short-account-ratio-contract-top-trader'
OKX_TAKER  = 'https://www.okx.com/api/v5/rubik/stat/taker-volume-contract'
OKX_DEPTH  = 'https://www.okx.com/api/v5/market/books'

# Bybit endpoints (public, no key)
BB_KLINE   = 'https://api.bybit.com/v5/market/kline'
BB_OI      = 'https://api.bybit.com/v5/market/open-interest'
BB_TAKER   = 'https://api.bybit.com/v5/market/account-ratio'


def _okx_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC-USDT-SWAP (OKX format)."""
    base = symbol.replace('USDT', '')
    return f'{base}-USDT-SWAP'


def _to_okx_interval(binance_interval: str) -> str:
    """Map Binance interval to OKX bar."""
    return {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1H', '4h': '4H', '1d': '1D',
    }.get(binance_interval, '1m')


def _to_bybit_interval(binance_interval: str) -> str:
    """Map Binance interval to Bybit numeric interval."""
    return {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '4h': '240', '1d': 'D',
    }.get(binance_interval, '1')


class MarketData:
    """Fetches market data with Binance→OKX fallback."""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
        
        # Track source per data type: {'klines': 'Binance', 'oi': 'OKX', ...}
        self._sources: Dict[str, str] = {}
        # Track errors for monitoring
        self._bn_errors: int = 0
        self._okx_errors: int = 0
        self._bb_errors: int = 0
    
    @property
    def sources(self) -> Dict[str, str]:
        return dict(self._sources)
    
    @property
    def source_summary(self) -> str:
        """Human-readable summary: 'Binance' or 'Binance+OKX' or 'OKX'."""
        vals = set(self._sources.values())
        if not vals:
            return '—'
        return '+'.join(sorted(vals))
    
    # ========================================
    # KLINES (taker buy/sell volumes)
    # ========================================
    
    def fetch_klines(self, symbol: str, limit: int = 60, interval: str = '1m') -> Optional[List[Dict]]:
        """Fetch klines with taker buy/sell. Returns [{p, v, b, s, h, l, o, t}, ...]
        
        interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d'  (Binance format)
        """
        # Map for OKX/Bybit
        okx_interval = _to_okx_interval(interval)
        bb_interval = _to_bybit_interval(interval)
        
        # Try Binance
        try:
            r = self._session.get(BN_KLINE,
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                candles = []
                for k in r.json():
                    try:
                        tv = float(k[7]); tb = float(k[10])
                        candles.append({'t': int(k[0]), 'p': float(k[4]), 'v': round(tv),
                                       'b': round(tb), 's': round(tv - tb),
                                       'h': float(k[2]), 'l': float(k[3]), 'o': float(k[1])})
                    except:
                        continue
                if candles:
                    self._sources['klines'] = 'Binance'
                    return candles
        except:
            self._bn_errors += 1
        
        # Fallback: OKX
        try:
            time.sleep(DELAY)
            okx_sym = _okx_symbol(symbol)
            r = self._session.get(OKX_KLINE,
                params={'instId': okx_sym, 'bar': okx_interval, 'limit': str(limit)},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                candles = []
                for k in data:
                    try:
                        tv = float(k[7])
                        o = float(k[1]); c = float(k[4])
                        buy_ratio = 0.6 if c > o else 0.4 if c < o else 0.5
                        tb = tv * buy_ratio
                        candles.append({'t': int(k[0]), 'p': c, 'v': round(tv),
                                       'b': round(tb), 's': round(tv - tb),
                                       'h': float(k[2]), 'l': float(k[3]), 'o': o})
                    except:
                        continue
                candles.reverse()
                if candles:
                    self._sources['klines'] = 'OKX'
                    return candles
        except:
            self._okx_errors += 1
        
        # Fallback 2: Bybit
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_KLINE,
                params={'category': 'linear', 'symbol': symbol, 'interval': bb_interval, 'limit': limit},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                result = r.json().get('result', {})
                data = result.get('list', [])
                candles = []
                for k in data:
                    try:
                        tv = float(k[6])
                        o = float(k[1]); c = float(k[4])
                        buy_ratio = 0.6 if c > o else 0.4 if c < o else 0.5
                        tb = tv * buy_ratio
                        candles.append({'t': int(k[0]), 'p': c, 'v': round(tv),
                                       'b': round(tb), 's': round(tv - tb),
                                       'h': float(k[2]), 'l': float(k[3]), 'o': o})
                    except:
                        continue
                candles.reverse()
                if candles:
                    self._sources['klines'] = 'Bybit'
                    return candles
        except:
            self._bb_errors += 1
        
        return None
    
    # ========================================
    # OPEN INTEREST
    # ========================================
    
    def fetch_oi(self, symbol: str, price: float = 0) -> Tuple[Optional[float], str]:
        """Fetch Open Interest in USD. Returns (oi_usd, source)."""
        # Binance
        try:
            r = self._session.get(BN_OI,
                params={'symbol': symbol}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                qty = float(r.json().get('openInterest', 0))
                oi_usd = qty * price if price else qty
                self._sources['oi'] = 'Binance'
                return oi_usd, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            okx_sym = _okx_symbol(symbol)
            r = self._session.get(OKX_OI,
                params={'instType': 'SWAP', 'ccy': symbol.replace('USDT', ''),
                        'period': '5m', 'begin': '', 'end': ''},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    oi_val = float(data[0].get('oi', 0))
                    oi_usd = oi_val * price if price else oi_val
                    self._sources['oi'] = 'OKX'
                    return oi_usd, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_OI,
                params={'category': 'linear', 'symbol': symbol, 'intervalTime': '5min', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                result = r.json().get('result', {})
                rows = result.get('list', [])
                if rows:
                    oi_val = float(rows[0].get('openInterest', 0))
                    oi_usd = oi_val * price if price else oi_val
                    self._sources['oi'] = 'Bybit'
                    return oi_usd, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # LONG/SHORT RATIO
    # ========================================
    
    def fetch_ls_ratio(self, symbol: str) -> Tuple[Optional[Dict], str]:
        """Fetch L/S ratio. Returns ({ls_ratio, ls_long}, source)."""
        # Binance
        try:
            r = self._session.get(BN_LS,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    result = {
                        'ls_ratio': float(latest.get('longShortRatio', 1)),
                        'ls_long': round(float(latest.get('longAccount', 0.5)) * 100, 1),
                    }
                    self._sources['ls'] = 'Binance'
                    return result, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_LS,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    ratio = float(data[0].get('ratio', 1))
                    long_pct = round(ratio / (1 + ratio) * 100, 1) if ratio > 0 else 50
                    result = {'ls_ratio': ratio, 'ls_long': long_pct}
                    self._sources['ls'] = 'OKX'
                    return result, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit (account ratio — closest to L/S)
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_TAKER,
                params={'category': 'linear', 'symbol': symbol, 'period': '1d', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                rows = r.json().get('result', {}).get('list', [])
                if rows:
                    buy_ratio = float(rows[0].get('buyRatio', 0.5))
                    sell_ratio = float(rows[0].get('sellRatio', 0.5))
                    ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
                    long_pct = round(buy_ratio * 100, 1)
                    result = {'ls_ratio': round(ratio, 3), 'ls_long': long_pct}
                    self._sources['ls'] = 'Bybit'
                    return result, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # TOP TRADER L/S RATIO
    # ========================================
    
    def fetch_top_ls(self, symbol: str) -> Tuple[Optional[Dict], str]:
        """Fetch Top Trader L/S. Returns ({top_ls, top_long}, source)."""
        # Binance
        try:
            r = self._session.get(BN_TOP_LS,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    result = {
                        'top_ls': float(latest.get('longShortRatio', 1)),
                        'top_long': round(float(latest.get('longAccount', 0.5)) * 100, 1),
                    }
                    self._sources['top_ls'] = 'Binance'
                    return result, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX (uses same endpoint with different params)
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_LS,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    ratio = float(data[0].get('ratio', 1))
                    long_pct = round(ratio / (1 + ratio) * 100, 1) if ratio > 0 else 50
                    result = {'top_ls': ratio, 'top_long': long_pct}
                    self._sources['top_ls'] = 'OKX'
                    return result, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit (uses same account-ratio as ls)
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_TAKER,
                params={'category': 'linear', 'symbol': symbol, 'period': '1d', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                rows = r.json().get('result', {}).get('list', [])
                if rows:
                    buy_ratio = float(rows[0].get('buyRatio', 0.5))
                    sell_ratio = float(rows[0].get('sellRatio', 0.5))
                    ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
                    long_pct = round(buy_ratio * 100, 1)
                    result = {'top_ls': round(ratio, 3), 'top_long': long_pct}
                    self._sources['top_ls'] = 'Bybit'
                    return result, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # TAKER BUY/SELL RATIO
    # ========================================
    
    def fetch_taker_ratio(self, symbol: str) -> Tuple[Optional[float], str]:
        """Fetch taker buy/sell ratio. Returns (ratio, source)."""
        # Binance
        try:
            r = self._session.get(BN_TAKER,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    ratio = float(data[-1].get('buySellRatio', 1))
                    self._sources['taker'] = 'Binance'
                    return ratio, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_TAKER,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    sell = float(data[0].get('sellVol', 1))
                    buy = float(data[0].get('buyVol', 1))
                    ratio = buy / sell if sell > 0 else 1
                    self._sources['taker'] = 'OKX'
                    return ratio, 'OKX'
        except:
            self._okx_errors += 1
        
        return None, ''
    
    # ========================================
    # FULL SENTIMENT (all 4 sentiment endpoints)
    # ========================================
    
    def fetch_sentiment(self, symbol: str) -> Dict:
        """Fetch OI + LS + TopLS + Taker with fallback. Returns dict with source info."""
        sent = {}
        sources = []
        
        # LS Ratio
        ls_data, ls_src = self.fetch_ls_ratio(symbol)
        if ls_data:
            sent.update(ls_data)
            if ls_src not in sources: sources.append(ls_src)
        time.sleep(DELAY)
        
        # Top Trader
        top_data, top_src = self.fetch_top_ls(symbol)
        if top_data:
            sent.update(top_data)
            if top_src not in sources: sources.append(top_src)
        time.sleep(DELAY)
        
        # Taker
        taker, taker_src = self.fetch_taker_ratio(symbol)
        if taker is not None:
            sent['taker'] = taker
            if taker_src not in sources: sources.append(taker_src)
        
        sent['_sources'] = '+'.join(sources) if sources else '—'
        return sent
    
    def get_status(self) -> Dict:
        return {
            'sources': self._sources,
            'summary': self.source_summary,
            'bn_errors': self._bn_errors,
            'okx_errors': self._okx_errors,
            'bb_errors': self._bb_errors,
        }


# Singleton
_instance: Optional[MarketData] = None

def get_market_data() -> MarketData:
    global _instance
    if _instance is None:
        _instance = MarketData()
    return _instance
