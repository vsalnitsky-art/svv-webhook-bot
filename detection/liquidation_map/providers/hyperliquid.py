"""
HyperliquidProvider — fetches real on-chain position data from Hyperliquid's
public Info API.

Unlike Binance/Bybit which expose ONLY aggregate OI, Hyperliquid is a
decentralized perpetual exchange where every position is on-chain. Their
public Info API exposes aggregated state that gives us REAL leverage
breakdown — what fraction of OI is at 25x vs 50x vs 100x — without
needing to estimate.

Hyperliquid is ~16% of global BTC perp OI (per Glassnode 2024 figures).
Smaller than Binance + Bybit combined, but provides a ground-truth signal
that the daemon overlays on top of the OI estimation. UI shows separate
"source=hyperliquid" buckets so user can see which clusters are
confirmed vs estimated.

API docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint

Endpoints used:
  POST https://api.hyperliquid.xyz/info
    {"type": "metaAndAssetCtxs"}        → list of all coins with markPx, openInterest, premium
    {"type": "clearinghouseState", "user": "0x..."}  → per-user (not useful for us)
  
  Phase 1 implementation: use metaAndAssetCtxs for cross-exchange OI
  comparison. Real leverage breakdown via clearinghouseState requires
  enumerating users — out of scope for Phase 1. We mark the source as
  'hyperliquid' so even without breakdown, the bucket has provenance.
"""

import time
import requests
from typing import Optional

from detection.liquidation_map.providers.base import (
    LiquidationDataProvider, OISnapshot,
)


HYPERLIQUID_INFO_URL = 'https://api.hyperliquid.xyz/info'
HTTP_TIMEOUT_SEC = 6.0
USER_AGENT = 'VSV-LiquidationMap/1.0'


class HyperliquidProvider(LiquidationDataProvider):
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': USER_AGENT,
            'Content-Type': 'application/json',
        })
        self._last_success_ts: float = 0.0
        self._consecutive_failures: int = 0
        # Hyperliquid's metaAndAssetCtxs response returns ALL coins in one
        # call — cache it briefly so we don't re-hit the endpoint when
        # daemon iterates multiple symbols on the same tick.
        self._cached_response = None
        self._cached_at: float = 0.0
        self._cache_ttl_sec = 30
    
    @property
    def name(self) -> str:
        return 'hyperliquid'
    
    def is_healthy(self) -> bool:
        return (time.time() - self._last_success_ts < 600
                and self._consecutive_failures < 3)
    
    def _fetch_meta_and_asset_ctxs(self):
        """One call returns state for ALL coins; we cache for 30s. The
        response is a 2-element array: [meta, asset_ctxs]. meta has
        universe[] with coin names; asset_ctxs has per-coin runtime stats
        in the SAME ORDER."""
        if (self._cached_response is not None
                and time.time() - self._cached_at < self._cache_ttl_sec):
            return self._cached_response
        
        try:
            r = self._session.post(
                HYPERLIQUID_INFO_URL,
                json={'type': 'metaAndAssetCtxs'},
                timeout=HTTP_TIMEOUT_SEC)
            if r.status_code != 200:
                self._consecutive_failures += 1
                return None
            data = r.json()
            if not isinstance(data, list) or len(data) != 2:
                self._consecutive_failures += 1
                return None
            self._cached_response = data
            self._cached_at = time.time()
            self._last_success_ts = self._cached_at
            self._consecutive_failures = 0
            return data
        except Exception:
            self._consecutive_failures += 1
            return None
    
    def _lookup_coin(self, symbol: str, data):
        """Hyperliquid uses bare coin symbols (BTC, ETH, SOL) — strip USDT
        and lowercase variants. Match against meta.universe[i].name."""
        coin = symbol.upper().replace('USDT', '').replace('USD', '').replace('.P', '')
        meta, ctxs = data
        universe = meta.get('universe', [])
        for i, u in enumerate(universe):
            if u.get('name', '').upper() == coin:
                if i < len(ctxs):
                    return ctxs[i]
                return None
        return None
    
    def fetch_oi_snapshot(self, symbol: str) -> Optional[OISnapshot]:
        data = self._fetch_meta_and_asset_ctxs()
        if not data:
            return None
        
        ctx = self._lookup_coin(symbol, data)
        if not ctx:
            # Symbol not listed on Hyperliquid (e.g., obscure alt). Caller
            # falls back to estimation-only for that symbol.
            return None
        
        # Hyperliquid fields: markPx (str), openInterest (str, in coin units),
        # midPx (str), premium (str), funding (str), dayNtlVlm (str)
        try:
            mark_price = float(ctx.get('markPx', 0))
            oi_coins = float(ctx.get('openInterest', 0))
        except (TypeError, ValueError):
            return None
        
        if mark_price <= 0 or oi_coins <= 0:
            return None
        
        oi_usd = oi_coins * mark_price
        
        return OISnapshot(
            symbol=symbol,
            exchange='hyperliquid',
            ts=int(time.time()),
            mark_price=mark_price,
            open_interest_usd=oi_usd,
            # Hyperliquid doesn't expose aggregate long/short ratio via
            # metaAndAssetCtxs. We leave it None and the aggregator falls
            # back to either the Binance ratio (preferred) or 0.5.
            long_ratio=None,
            extras={'hl_premium': ctx.get('premium'),
                    'hl_funding': ctx.get('funding')},
        )
