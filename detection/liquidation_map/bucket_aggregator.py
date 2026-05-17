"""
bucket_aggregator.py — converts OI deltas between snapshots into price-bucket
contributions, distributed across leverage tiers and long/short sides.

This is the core of the estimation algorithm. Given:
  prev snapshot  — OI=X, mark=P0   at time t-1
  curr snapshot  — OI=Y, mark=P1   at time t

we compute:
  delta_oi_usd = (Y - X)            # total notional opened (or closed) this tick
  long_share   = curr.long_ratio    # from Binance topLongShortPositionRatio
                                     # OR derived from price movement
                                     # OR fallback 0.5

For each leverage tier in DEFAULT_LEVERAGE_WEIGHTS:
  long_notional  = delta_oi_usd × weight × long_share
  short_notional = delta_oi_usd × weight × (1 - long_share)
  
  For LONG positions opened: liq_price computed via real Binance formula
  For SHORT positions opened: same
  
  Each liq_price is then quantized to a bucket (e.g. $25 on BTC) and
  emitted as a BucketContribution. Daemon writes these to the DB.

When delta_oi_usd is NEGATIVE (positions closed/liquidated): we don't
generate new buckets. The mitigation pass (separate) handles existing
buckets when price wicks through them.

The function is pure and stateless — easy to unit-test.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Dict

from detection.liquidation_map.liquidation_math import (
    liquidation_price, DEFAULT_LEVERAGE_WEIGHTS,
)


@dataclass
class BucketContribution:
    """One emit from the aggregator — a slice of OI delta attributed to a
    specific (price-bucket, side, leverage) triplet."""
    bucket_price: float
    side: str           # 'long' or 'short'
    leverage: int       # 25, 50, 100
    usd_added: float    # how much notional landed in this bucket
    source: str         # 'estimation' (default) or 'hyperliquid' for verified


# Default bucket size in dollars. Override per symbol if needed.
def default_bucket_size(symbol: str, mark_price: float) -> float:
    """Bucket size scaling: ~0.03% of price for everything. Gives ~$25 on
    BTC at $78k, ~$0.90 on ETH at $3k, ~$0.05 on SOL at $150. Empirically
    matches what CoinGlass shows at zoomed-in 24h view."""
    s = symbol.upper()
    if s.startswith('BTC'):
        return 25.0
    if s.startswith('ETH'):
        return 1.0
    # Generic: 0.03% of mark price, with sensible floor
    return max(round(mark_price * 0.0003, 4), 0.0001)


def quantize_to_bucket(price: float, bucket_size: float) -> float:
    """Round price DOWN to the bucket boundary so adjacent positions
    collapse onto the same row. We return the bucket MIDPOINT (boundary +
    bucket_size/2) so UI can plot bands centered correctly."""
    if bucket_size <= 0:
        return price
    floor_price = (price // bucket_size) * bucket_size
    return floor_price + bucket_size / 2


def derive_long_ratio_from_price_move(prev_mark: float,
                                         curr_mark: float) -> float:
    """When provider didn't give us a long/short ratio, infer one from
    the price movement during the OI accumulation period.
    
    Heuristic: OI grew while price rose → more longs piled in. OI grew
    while price fell → more shorts. The magnitude of the ratio shift is
    capped to avoid extreme outputs from tiny noise moves.
    
    Returns long_share in [0.3, 0.7]. Never returns 0 or 1 because no
    OI delta is truly 100% one-sided in practice.
    """
    if prev_mark <= 0 or curr_mark <= 0:
        return 0.5
    pct_change = (curr_mark - prev_mark) / prev_mark
    # Cap influence at ±1% move = full bias swing
    capped = max(-0.01, min(0.01, pct_change))
    # 1% up → long_share = 0.7; 1% down → 0.3; 0% → 0.5
    return 0.5 + (capped / 0.01) * 0.2


def attribute_oi_delta(
    prev_snapshot,
    curr_snapshot,
    *,
    symbol: str,
    leverage_weights: Optional[Dict[int, float]] = None,
    bucket_size: Optional[float] = None,
    source: str = 'estimation',
) -> Iterable[BucketContribution]:
    """Emit BucketContribution objects for the OI delta between snapshots.
    
    Args:
      prev_snapshot, curr_snapshot: OISnapshot from the same provider/symbol
      symbol: ticker, used for tier table + bucket sizing
      leverage_weights: override default 40/35/25 if needed (e.g., from
                        Hyperliquid's real breakdown when we have it)
      bucket_size: override the default scaler
      source: tag stamped on every emitted contribution
    
    Yields one BucketContribution per (leverage tier × side) combination
    when delta > 0, nothing when delta ≤ 0.
    """
    if prev_snapshot is None or curr_snapshot is None:
        return
    if curr_snapshot.ts <= prev_snapshot.ts:
        return  # out-of-order or duplicate
    
    delta_usd = curr_snapshot.open_interest_usd - prev_snapshot.open_interest_usd
    if delta_usd <= 0:
        # Net close — no new buckets. Mitigation pass handles existing.
        return
    
    weights = leverage_weights or DEFAULT_LEVERAGE_WEIGHTS
    bsize = bucket_size or default_bucket_size(symbol, curr_snapshot.mark_price)
    
    # Long/short attribution priority:
    #   1. Provider's long_ratio (Binance topLongShortPositionRatio) if present
    #   2. Price-move heuristic
    #   3. Fallback 0.5
    if curr_snapshot.long_ratio is not None:
        long_share = max(0.0, min(1.0, curr_snapshot.long_ratio))
    else:
        long_share = derive_long_ratio_from_price_move(
            prev_snapshot.mark_price, curr_snapshot.mark_price)
    short_share = 1.0 - long_share
    
    # New positions opened at approximately the current mark price.
    # This is the standard assumption everyone uses — we don't have the
    # actual entry distribution.
    entry = curr_snapshot.mark_price
    
    for lev, w in weights.items():
        tier_usd = delta_usd * w
        if tier_usd <= 0:
            continue
        
        long_notional = tier_usd * long_share
        short_notional = tier_usd * short_share
        
        if long_notional > 0:
            liq_long = liquidation_price(
                'LONG', entry, lev, long_notional, symbol=symbol)
            yield BucketContribution(
                bucket_price=quantize_to_bucket(liq_long, bsize),
                side='long', leverage=lev,
                usd_added=long_notional, source=source,
            )
        
        if short_notional > 0:
            liq_short = liquidation_price(
                'SHORT', entry, lev, short_notional, symbol=symbol)
            yield BucketContribution(
                bucket_price=quantize_to_bucket(liq_short, bsize),
                side='short', leverage=lev,
                usd_added=short_notional, source=source,
            )


# ============================================================================
# Cluster detection — "magnet zones" identified for the UI summary
# ============================================================================

def find_clusters(buckets: list, mark_price: float,
                    min_total_usd: float = 1_000_000,
                    proximity_pct: float = 0.005) -> list:
    """Group adjacent buckets (within proximity_pct of each other) into
    cluster zones. Returns list of {price_low, price_high, total_usd, side,
    magnet_score} dicts sorted by total_usd DESC.
    
    magnet_score is a 0..1 normalized intensity reflecting how strongly
    this cluster might attract price — bigger cluster, closer to current
    price, more leverage all push the score up.
    """
    if not buckets:
        return []
    
    # Sort buckets by price ASC (caller may have already done this)
    sorted_b = sorted(buckets, key=lambda b: b['bucket_price'])
    
    clusters = []
    current = None
    for b in sorted_b:
        bp = b['bucket_price']
        side = b['side']
        if current is None:
            current = {
                'side': side, 'price_low': bp, 'price_high': bp,
                'total_usd': b['cumulative_usd'],
                'leverage_mix': {b['leverage']: b['cumulative_usd']},
            }
            continue
        # Grow current cluster if same side AND price within proximity_pct
        gap_pct = (bp - current['price_high']) / mark_price
        if side == current['side'] and gap_pct <= proximity_pct:
            current['price_high'] = bp
            current['total_usd'] += b['cumulative_usd']
            current['leverage_mix'][b['leverage']] = \
                current['leverage_mix'].get(b['leverage'], 0) + b['cumulative_usd']
        else:
            if current['total_usd'] >= min_total_usd:
                clusters.append(current)
            current = {
                'side': side, 'price_low': bp, 'price_high': bp,
                'total_usd': b['cumulative_usd'],
                'leverage_mix': {b['leverage']: b['cumulative_usd']},
            }
    if current and current['total_usd'] >= min_total_usd:
        clusters.append(current)
    
    # Magnet score — relative to the heaviest cluster found
    max_usd = max((c['total_usd'] for c in clusters), default=0)
    for c in clusters:
        # Volume component (0..1)
        vol_score = c['total_usd'] / max_usd if max_usd else 0
        # Proximity component — closer to current price = stronger magnet
        midpoint = (c['price_low'] + c['price_high']) / 2
        dist_pct = abs(midpoint - mark_price) / mark_price
        # 0% away = 1.0; 10% away = 0.0; linear
        prox_score = max(0, 1 - dist_pct / 0.10)
        # Weighted blend — volume matters more than proximity
        c['magnet_score'] = round(0.7 * vol_score + 0.3 * prox_score, 3)
    
    clusters.sort(key=lambda c: c['total_usd'], reverse=True)
    return clusters
