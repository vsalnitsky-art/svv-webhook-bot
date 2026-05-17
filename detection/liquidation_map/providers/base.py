"""
Abstract base class for liquidation data providers.

Each provider has ONE job: fetch the current state of open interest (and
optionally long/short positioning) for a given symbol on a given exchange.
The bucket aggregator combines snapshots over time into actual liquidation
levels — providers know nothing about leverage tiers, margin formulas, or
buckets. They only know how to talk to their data source.

Adding a new provider (e.g., CoinGlass premium API) is a single new file
implementing this interface and registering itself with the daemon's
provider list. No other code needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class OISnapshot:
    """One open-interest snapshot from one provider at one moment.
    
    open_interest_usd is the TOTAL notional value of open positions in USD.
    For Binance Futures /fapi/v1/openInterest the raw response is in
    contracts/coins — the provider does the conversion via mark price.
    
    long_ratio (0.0–1.0) is optional. Some providers expose top-trader
    long/short ratios (Binance /futures/data/topLongShortPositionRatio),
    others don't. None means "fall back to 0.5".
    """
    symbol: str
    exchange: str
    ts: int                          # epoch seconds, when the data was fetched
    mark_price: float
    open_interest_usd: float
    long_ratio: Optional[float] = None
    # Provider-specific extras (used by Hyperliquid to pass through actual
    # leverage breakdowns when available). Daemon ignores this for providers
    # that don't fill it.
    extras: dict = None


@dataclass
class HyperliquidPositionBreakdown:
    """Optional per-leverage-tier OI delta from Hyperliquid clearinghouseState.
    
    Unlike OISnapshot which is aggregated, this is a structured breakdown
    of how much new OI opened at each leverage tier — REAL DATA, not
    estimation. Daemon uses this directly to attribute buckets without
    going through DEFAULT_LEVERAGE_WEIGHTS.
    """
    symbol: str
    ts: int
    mark_price: float
    # leverage_tier → (long_usd_delta, short_usd_delta) since previous snap
    breakdown: dict   # {25: (l, s), 50: (l, s), 100: (l, s)}


class LiquidationDataProvider(ABC):
    """Interface — every provider must implement fetch_oi_snapshot().
    fetch_position_breakdown() is optional (returns None) for providers
    that only expose aggregate OI."""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def fetch_oi_snapshot(self, symbol: str) -> Optional[OISnapshot]:
        """Return current OI + mark price for symbol, or None on failure.
        Must not raise — provider's job to swallow network errors so
        daemon can degrade gracefully when one source is down."""
    
    def fetch_position_breakdown(self, symbol: str) -> Optional[HyperliquidPositionBreakdown]:
        """Override only if provider exposes per-leverage-tier real data
        (Hyperliquid does). Default returns None — caller falls back to
        weighted estimation."""
        return None
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick boolean: has this provider been returning data recently?
        Surfaced to the UI as a status indicator so user can see when
        Hyperliquid is down and the map is running on estimation only."""
