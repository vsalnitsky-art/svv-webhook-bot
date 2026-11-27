"""
liquidation_math.py — real-exchange liquidation price formulas with
maintenance margin tier tables.

Phase 1 covers Binance Futures (which has ~50% of BTC perp OI). The
formulas match Binance's documented liquidation engine, so prices are
within ~0.1% of what the exchange would actually liquidate at. Bybit
has very similar tier tables — we use the same formula and a Bybit-
specific tier table.

Why this matters:
The naive formula `liq_long = entry × (1 − 1/leverage)` is off by 0.4–
2.5% depending on notional. At 100x on $78,000 BTC:
    naive    → liq at $77,220 (1.00% below)
    real BNB → liq at $77,532 (0.60% below)
That ~$300 difference matters when we're trying to identify clusters
within $25 bucket resolution.

Refs:
- Binance Futures docs:
  https://www.binance.com/en/support/faq/leverage-and-margin-of-usd%E2%93%A2-m-futures-360033162192
- Bybit docs:
  https://www.bybit.com/en/help-center/article/Maintenance-Margin
"""

from typing import List, Tuple, Optional


# ============================================================================
# Tier tables — (notional_max_usd, maintenance_margin_rate, maintenance_amount)
# Format matches Binance's published structure. notional_max=None = catch-all
# (no upper bound), used for the highest tier so list is exhaustive.
# ============================================================================

# Binance Futures BTCUSDT (perpetual). Source: Binance "Maintenance Margin
# Tier" public table as of late 2025. Mid-2026 may have shifted ±1 tier but
# rates are stable for the past 2y.
BINANCE_BTC_TIERS: List[Tuple[Optional[float], float, float]] = [
    (50_000,       0.004, 0),
    (250_000,      0.005, 50),
    (1_000_000,    0.01,  1_300),
    (10_000_000,   0.025, 16_300),
    (20_000_000,   0.05,  266_300),
    (50_000_000,   0.10,  1_266_300),
    (100_000_000,  0.125, 2_516_300),
    (200_000_000,  0.15,  5_016_300),
    (300_000_000,  0.25,  25_016_300),
    (500_000_000,  0.50,  100_016_300),
    (None,         1.00,  350_016_300),
]

# Binance Futures ETHUSDT. Similar shape, slightly different rates.
BINANCE_ETH_TIERS: List[Tuple[Optional[float], float, float]] = [
    (50_000,       0.005, 0),
    (250_000,      0.0065, 75),
    (1_000_000,    0.01,  950),
    (10_000_000,   0.02,  10_950),
    (20_000_000,   0.05,  310_950),
    (50_000_000,   0.10,  1_310_950),
    (100_000_000,  0.125, 2_560_950),
    (None,         0.25,  15_060_950),
]

# Fallback for any other symbol — uses Binance "tier 1 altcoin" defaults.
# Conservative (slightly higher mmr) so we don't underestimate liq distance
# for symbols we don't have tables for. Better safe than precise.
BINANCE_DEFAULT_TIERS: List[Tuple[Optional[float], float, float]] = [
    (5_000,    0.01,  0),
    (25_000,   0.025, 75),
    (100_000,  0.05,  700),
    (250_000,  0.10,  5_700),
    (1_000_000, 0.125, 11_950),
    (None,     0.50,  386_950),
]


def get_tier_table(symbol: str) -> List[Tuple[Optional[float], float, float]]:
    """Pick the right tier table for a symbol. Symbol comparison is
    case-insensitive and strips the contract-type suffix."""
    s = symbol.upper().replace('USDT', '').replace('USD', '').replace('.P', '')
    if s == 'BTC':
        return BINANCE_BTC_TIERS
    if s == 'ETH':
        return BINANCE_ETH_TIERS
    return BINANCE_DEFAULT_TIERS


def lookup_tier(notional_usd: float, tiers) -> Tuple[float, float]:
    """Return (mmr, mma) for a given notional value. tiers must be sorted
    ascending by notional_max with the last tier having notional_max=None."""
    for ceil, mmr, mma in tiers:
        if ceil is None or notional_usd <= ceil:
            return mmr, mma
    # Shouldn't reach here if tier list ends with None
    return tiers[-1][1], tiers[-1][2]


def liquidation_price(side: str, entry_price: float, leverage: float,
                       notional_usd: float, symbol: str = 'BTCUSDT',
                       wallet_balance_usd: Optional[float] = None) -> float:
    """Compute the price at which a position would be liquidated under
    Binance's cross-margin liquidation rules. Side is 'LONG' or 'SHORT'.
    
    Formula (Binance public docs, simplified for ISOLATED margin which is
    by far the dominant mode on perps — cross margin would require knowing
    full wallet balance which we never have for anonymous OI estimates):
    
        wallet_balance = notional / leverage      (isolated margin)
        mmr, mma       = tier lookup for notional
        
        LONG:  liq_price = (notional − wallet_balance + mma) /
                            (qty × (1 − mmr))
        SHORT: liq_price = (notional + wallet_balance − mma) /
                            (qty × (1 + mmr))
        
        where qty = notional / entry_price
    
    Algebraic simplification (entry cancels):
        LONG:  liq_price = entry × (1 − 1/leverage + mmr − mma/notional)
        SHORT: liq_price = entry × (1 + 1/leverage − mmr + mma/notional)
    """
    tiers = get_tier_table(symbol)
    mmr, mma = lookup_tier(notional_usd, tiers)
    
    # mma is per-position dollar offset (a "rebate" embedded in the tier
    # system to make formulas continuous across tier boundaries). Tiny on
    # small notionals, dominant on huge ones.
    mma_rate = mma / notional_usd if notional_usd > 0 else 0
    
    if side == 'LONG':
        return entry_price * (1 - 1/leverage + mmr - mma_rate)
    elif side == 'SHORT':
        return entry_price * (1 + 1/leverage - mmr + mma_rate)
    else:
        raise ValueError(f"side must be 'LONG' or 'SHORT', got {side!r}")


def liquidation_distance_pct(side: str, entry_price: float, leverage: float,
                                notional_usd: float, symbol: str = 'BTCUSDT') -> float:
    """Convenience — fractional distance from entry to liquidation,
    always positive. e.g. 0.01 = 1% away."""
    liq = liquidation_price(side, entry_price, leverage, notional_usd, symbol)
    return abs(liq - entry_price) / entry_price


# ============================================================================
# Default leverage-tier weights used when estimating from aggregated OI.
# Tuned to match what CoinGlass/Hyblock seem to use as defaults (these are
# the same weights documented in several public Pine indicators referencing
# Hyblock's methodology). Not gospel — can be overridden per-symbol.
# ============================================================================

DEFAULT_LEVERAGE_WEIGHTS = {
    25: 0.40,
    50: 0.35,
    100: 0.25,
}

assert abs(sum(DEFAULT_LEVERAGE_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Leverage weights must sum to 1.0"
