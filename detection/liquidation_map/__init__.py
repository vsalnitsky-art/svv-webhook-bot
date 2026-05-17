"""
Liquidation Map module — estimates and tracks liquidation level clusters
for perpetual futures symbols across exchanges. Built on top of:

  - Aggregated OI deltas from Binance + Bybit (estimation provider)
  - Real on-chain position data from Hyperliquid (truth provider, ~16% OI)
  - Real Binance/Bybit maintenance margin tier tables (accurate prices)

Tier 2 architecture per the user's choice — free, with Hyperliquid's
clearinghouseState as a ground-truth overlay on the estimation. The
data provider layer is pluggable; adding CoinGlass later is a single
new file under providers/.
"""

from detection.liquidation_map.liquidation_map import (
    LiquidationMapDaemon,
    init_liquidation_map,
    get_liquidation_map,
)

__all__ = [
    'LiquidationMapDaemon',
    'init_liquidation_map',
    'get_liquidation_map',
]
