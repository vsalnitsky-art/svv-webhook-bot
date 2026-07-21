"""
OIEstimationProvider — fetches aggregated OI from Binance Futures and
Bybit V5, plus Binance's topLongShortPositionRatio for long/short bias.

Aggregated means we get TOTAL OI without per-leverage breakdown — the
daemon will attribute the delta to leverage tiers using fixed weights
(see liquidation_math.DEFAULT_LEVERAGE_WEIGHTS). This is the same
methodology used by most public liquidation-map indicators including
the Alien_Algorithms TradingView script the user wants to mirror.

We poll three Binance endpoints and one Bybit endpoint, with timeouts
short enough that one slow endpoint never blocks a scan tick for more
than a few seconds. Each endpoint is best-effort — if one fails we
still return a usable OISnapshot built from whatever did succeed.
"""

import os
import time
import requests
from typing import Optional

# Binance блокує IP дата-центрів Render (HTTP 418), тож на проді ці запити й так
# падають у таймаут щотіка × кожен символ. За замовчуванням Binance ВИМКНЕНО —
# OI рахується лише по Bybit (результат той самий, що й зараз, бо Binance і так
# недоступний), але без марних 3 запитів на символ. Повернути: LIQMAP_USE_BINANCE=1.
_USE_BINANCE = os.getenv('LIQMAP_USE_BINANCE', '0').lower() in ('1', 'true', 'yes', 'on')

from detection.liquidation_map.providers.base import (
    LiquidationDataProvider, OISnapshot,
)


# ============================================================================
# Endpoint configuration. All public, no auth required, generous rate limits.
# ============================================================================

BINANCE_FAPI = 'https://fapi.binance.com'
BYBIT_V5 = 'https://api.bybit.com'

# Hard caps — providers must never block daemon for long
HTTP_TIMEOUT_SEC = 6.0
USER_AGENT = 'VSV-LiquidationMap/1.0'


class OIEstimationProvider(LiquidationDataProvider):
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': USER_AGENT})
        self._last_success_ts: float = 0.0
        self._consecutive_failures: int = 0
    
    @property
    def name(self) -> str:
        return 'estimation'
    
    def is_healthy(self) -> bool:
        # Healthy = last success within the last 5 minutes AND not in a
        # consecutive-failure spiral
        return (time.time() - self._last_success_ts < 300
                and self._consecutive_failures < 3)
    
    # ----- Binance fetches -----
    
    def _binance_open_interest(self, symbol: str) -> Optional[float]:
        """Returns OI in CONTRACTS (coin units), not USD. Caller multiplies
        by mark_price to get notional."""
        try:
            r = self._session.get(
                f'{BINANCE_FAPI}/fapi/v1/openInterest',
                params={'symbol': symbol}, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            return float(r.json()['openInterest'])
        except Exception:
            return None
    
    def _binance_mark_price(self, symbol: str) -> Optional[float]:
        try:
            r = self._session.get(
                f'{BINANCE_FAPI}/fapi/v1/premiumIndex',
                params={'symbol': symbol}, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            return float(r.json()['markPrice'])
        except Exception:
            return None
    
    def _binance_long_short_ratio(self, symbol: str) -> Optional[float]:
        """topLongShortPositionRatio — gives ratio of top-trader long/short
        positions on Binance. Returns the LONG share (0..1). This is a
        much better proxy than splitting OI 50/50."""
        try:
            r = self._session.get(
                f'{BINANCE_FAPI}/futures/data/topLongShortPositionRatio',
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data:
                return None
            # Endpoint returns longAccount/shortAccount as strings 0..1 each;
            # OR longShortRatio (e.g. "1.45" = 1.45 longs per short).
            row = data[0]
            if 'longAccount' in row:
                return float(row['longAccount'])
            if 'longShortRatio' in row:
                ratio = float(row['longShortRatio'])
                # ratio = long/short; long_share = ratio/(ratio+1)
                return ratio / (ratio + 1) if ratio > 0 else None
            return None
        except Exception:
            return None
    
    # ----- Bybit fetches -----
    
    def _bybit_open_interest(self, symbol: str) -> Optional[dict]:
        """Returns dict with 'oi_contracts' and 'mark_price'. Bybit V5 OI
        endpoint requires intervalTime — we use 5min, the smallest available."""
        try:
            r = self._session.get(
                f'{BYBIT_V5}/v5/market/open-interest',
                params={'category': 'linear', 'symbol': symbol,
                        'intervalTime': '5min', 'limit': 1},
                timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            j = r.json()
            if j.get('retCode') != 0:
                return None
            lst = j.get('result', {}).get('list', [])
            if not lst:
                return None
            return {'oi_contracts': float(lst[0]['openInterest'])}
        except Exception:
            return None
    
    def _bybit_mark_price(self, symbol: str) -> Optional[float]:
        try:
            r = self._session.get(
                f'{BYBIT_V5}/v5/market/tickers',
                params={'category': 'linear', 'symbol': symbol},
                timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                return None
            j = r.json()
            if j.get('retCode') != 0:
                return None
            lst = j.get('result', {}).get('list', [])
            if not lst:
                return None
            return float(lst[0]['markPrice'])
        except Exception:
            return None
    
    # ----- Combined snapshot -----
    
    def fetch_oi_snapshot(self, symbol: str) -> Optional[OISnapshot]:
        """Aggregate Binance + Bybit OI into a single snapshot. We sum the
        notional values across exchanges to get a market-wide estimate.
        Long ratio comes from Binance only — Bybit doesn't expose a
        comparable public endpoint.
        
        Returns None only when BOTH exchanges fail completely. As long as
        one source returns data we produce a partial snapshot."""
        ts = int(time.time())

        # Binance — вимкнено за замовчуванням (заблоковано на Render, і так падає).
        # Повернути через env LIQMAP_USE_BINANCE=1. Без нього — 0 марних запитів.
        if _USE_BINANCE:
            bnb_price = self._binance_mark_price(symbol)
            bnb_oi_contracts = self._binance_open_interest(symbol)
            bnb_oi_usd = (bnb_oi_contracts * bnb_price
                           if bnb_oi_contracts and bnb_price else 0)
            # Long/short ratio — Binance only
            long_ratio = self._binance_long_short_ratio(symbol)
        else:
            bnb_price = None
            bnb_oi_usd = 0.0
            long_ratio = None

        # Bybit — same approach
        bybit_data = self._bybit_open_interest(symbol)
        bybit_price = self._bybit_mark_price(symbol)
        bybit_oi_usd = 0.0
        if bybit_data and bybit_price:
            bybit_oi_usd = bybit_data['oi_contracts'] * bybit_price
        
        total_oi_usd = bnb_oi_usd + bybit_oi_usd
        if total_oi_usd <= 0:
            self._consecutive_failures += 1
            return None
        
        # Prefer Binance mark price (deeper liquidity), fall back to Bybit
        mark_price = bnb_price or bybit_price
        if not mark_price:
            self._consecutive_failures += 1
            return None
        
        self._last_success_ts = time.time()
        self._consecutive_failures = 0
        
        return OISnapshot(
            symbol=symbol,
            exchange='aggregated',
            ts=ts,
            mark_price=mark_price,
            open_interest_usd=total_oi_usd,
            long_ratio=long_ratio,
            extras={
                'binance_oi_usd': bnb_oi_usd,
                'bybit_oi_usd': bybit_oi_usd,
                'binance_mark': bnb_price,
                'bybit_mark': bybit_price,
            },
        )
