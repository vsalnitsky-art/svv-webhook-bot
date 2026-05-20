"""CoinGecko client for market-cap-based universe filtering.

Purpose: the Volumized OB Radar's default universe is "top-N USDT-perps by
24h volume" which is dominated by memcoins and freshly-pumped tokens. This
client provides an alternative ranking source — CoinGecko market cap rank —
so the radar can instead scan "top-N coins by market cap".

Design notes / honest caveats baked in:

  * Symbol mapping is the hard part. CoinGecko uses ids like 'bitcoin' and
    tickers like 'btc'; exchanges use 'BTCUSDT'. We map exchange perp symbol
    -> base ticker -> CoinGecko coin (highest market cap wins on ticker
    collisions). Exchange 1000x multiplier prefixes (1000PEPE, 1000SHIB) are
    stripped before matching.

  * Market cap rank changes slowly, so we cache the full top-N snapshot for
    several hours instead of hitting the API every scan. One '/coins/markets'
    call with per_page=250 covers the whole top-250.

  * Demo API key via env COINGECKO_API_KEY (header x-cg-demo-api-key on
    api.coingecko.com). Falls back to keyless (heavily rate-limited) if not
    set — but keyless is unreliable, so a key is strongly recommended.

  * Not every perp has a CoinGecko market cap (synthetics like XAUT, brand
    new listings). Those simply won't pass a market-cap filter — expected.
"""

import os
import time
import requests
from typing import Dict, List, Optional

CG_DEMO_BASE = 'https://api.coingecko.com/api/v3'
CG_PRO_BASE = 'https://pro-api.coingecko.com/api/v3'
REQUEST_TIMEOUT = 15

# Cache TTL — market cap rank moves slowly, refetch every 6 hours.
CACHE_TTL_SECS = 6 * 3600

# Exchange multiplier prefixes: "1000PEPEUSDT" trades 1000x PEPE; the
# underlying coin on CoinGecko is just PEPE. Strip these for matching.
MULTIPLIER_PREFIXES = ('1000000', '10000', '1000', '100')

# Manual ticker -> coingecko_id overrides for known ambiguous / special
# cases where the auto "highest-market-cap-for-ticker" heuristic picks wrong.
# Extend as needed; this is the documented escape hatch for mapping edge cases.
TICKER_OVERRIDES = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'DOGE': 'dogecoin',
    'TON': 'the-open-network',
    'ADA': 'cardano',
    'AVAX': 'avalanche-2',
    'LINK': 'chainlink',
    'DOT': 'polkadot',
    'MATIC': 'matic-network',
    'POL': 'polygon-ecosystem-token',
    'NEAR': 'near',
    'LTC': 'litecoin',
    'UNI': 'uniswap',
    'PEPE': 'pepe',
    'SHIB': 'shiba-inu',
    'WIF': 'dogwifcoin',
    'BONK': 'bonk',
    'ARB': 'arbitrum',
    'OP': 'optimism',
    'INJ': 'injective-protocol',
    'SUI': 'sui',
    'SEI': 'sei-network',
    'TIA': 'celestia',
    'APT': 'aptos',
    'RNDR': 'render-token',
    'RENDER': 'render-token',
    'FET': 'fetch-ai',
    'WLD': 'worldcoin-wld',
    'ENA': 'ethena',
    'ONDO': 'ondo-finance',
    'JUP': 'jupiter-exchange-solana',
    'TAO': 'bittensor',
    'HYPE': 'hyperliquid',
    'LDO': 'lido-dao',
    'ZEC': 'zcash',
    'NEO': 'neo',
    'AXS': 'axie-infinity',
    'ORDI': 'ordinals',
}


class CoinGeckoClient:
    """Fetches and caches CoinGecko market-cap rankings, and maps exchange
    perp symbols to those rankings."""

    def __init__(self, api_key: Optional[str] = None):
        # Prefer explicit arg, then env. Demo key uses api.coingecko.com.
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY', '').strip()
        self._session = requests.Session()
        # In-memory cache of the latest markets snapshot
        self._markets_cache: List[Dict] = []     # raw coin dicts
        self._ticker_to_rank: Dict[str, int] = {}  # 'BTC' -> 1
        self._id_to_rank: Dict[str, int] = {}      # 'bitcoin' -> 1
        self._cache_at: float = 0.0

    def _headers(self) -> Dict[str, str]:
        if self.api_key:
            return {'x-cg-demo-api-key': self.api_key, 'accept': 'application/json'}
        return {'accept': 'application/json'}

    def _fetch_markets(self, per_page: int = 250) -> Optional[List[Dict]]:
        """Fetch top coins by market cap. Returns raw CoinGecko coin dicts
        with id, symbol, market_cap_rank, market_cap. None on failure."""
        url = f'{CG_DEMO_BASE}/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': min(250, per_page),
            'page': 1,
            'sparkline': 'false',
        }
        try:
            r = self._session.get(url, params=params, headers=self._headers(),
                                  timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                print('[CoinGecko] Rate limited (429) — using cached data if available')
                return None
            if r.status_code != 200:
                print(f'[CoinGecko] markets fetch HTTP {r.status_code}')
                return None
            data = r.json()
            if not isinstance(data, list):
                return None
            return data
        except Exception as e:
            print(f'[CoinGecko] markets fetch error: {e}')
            return None

    def refresh(self, force: bool = False) -> bool:
        """Refresh the market cap snapshot if cache is stale (or force=True).
        Builds ticker->rank and id->rank maps. Returns True if data is
        available (fresh or cached), False if no data at all."""
        fresh = (time.time() - self._cache_at) < CACHE_TTL_SECS
        if fresh and not force and self._markets_cache:
            return True

        data = self._fetch_markets(per_page=250)
        if not data:
            # Keep old cache if we have one — better stale than nothing
            return bool(self._markets_cache)

        ticker_to_rank: Dict[str, int] = {}
        id_to_rank: Dict[str, int] = {}
        for coin in data:
            rank = coin.get('market_cap_rank')
            if rank is None:
                continue
            cid = (coin.get('id') or '').lower()
            ticker = (coin.get('symbol') or '').upper()
            if cid:
                id_to_rank[cid] = rank
            # On ticker collision, keep the BEST (lowest) rank — i.e. the
            # highest-market-cap coin that uses that ticker.
            if ticker:
                if ticker not in ticker_to_rank or rank < ticker_to_rank[ticker]:
                    ticker_to_rank[ticker] = rank

        self._markets_cache = data
        self._ticker_to_rank = ticker_to_rank
        self._id_to_rank = id_to_rank
        self._cache_at = time.time()
        print(f'[CoinGecko] Refreshed market cap snapshot: '
              f'{len(data)} coins, {len(ticker_to_rank)} tickers mapped')
        return True

    @staticmethod
    def _base_ticker(perp_symbol: str) -> str:
        """Extract the base ticker from an exchange perp symbol.
        'BTCUSDT' -> 'BTC', '1000PEPEUSDT' -> 'PEPE', '1000000MOGUSDT' -> 'MOG'.
        """
        s = perp_symbol.upper()
        # Strip the USDT quote suffix
        if s.endswith('USDT'):
            s = s[:-4]
        elif s.endswith('USD'):
            s = s[:-3]
        # Strip exchange multiplier prefixes (longest first to avoid partial)
        for pref in MULTIPLIER_PREFIXES:
            if s.startswith(pref) and len(s) > len(pref):
                s = s[len(pref):]
                break
        return s

    def get_rank(self, perp_symbol: str) -> Optional[int]:
        """Return the CoinGecko market cap rank for an exchange perp symbol,
        or None if the coin isn't found / has no market cap.

        Resolution order:
          1. Base ticker -> TICKER_OVERRIDES -> id_to_rank (most reliable)
          2. Base ticker -> ticker_to_rank (auto, highest-cap on collision)
        """
        base = self._base_ticker(perp_symbol)
        # 1. Explicit override -> id
        cid = TICKER_OVERRIDES.get(base)
        if cid and cid in self._id_to_rank:
            return self._id_to_rank[cid]
        # 2. Auto ticker map
        return self._ticker_to_rank.get(base)

    def filter_by_marketcap(self, perp_symbols: List[str],
                             max_rank: int = 150) -> List[Dict]:
        """Given a list of exchange perp symbols, return those whose
        CoinGecko market cap rank is <= max_rank, each annotated with its
        rank, sorted by rank ascending (highest market cap first).

        Symbols with no resolvable market cap rank are dropped.
        """
        self.refresh()
        out = []
        for sym in perp_symbols:
            rank = self.get_rank(sym)
            if rank is not None and rank <= max_rank:
                out.append({'symbol': sym, 'market_cap_rank': rank})
        out.sort(key=lambda x: x['market_cap_rank'])
        return out

    def is_available(self) -> bool:
        """True if we have (cached or fresh) ranking data to work with."""
        return bool(self._markets_cache) or self.refresh()


# Singleton accessor
_client: Optional[CoinGeckoClient] = None


def get_coingecko_client() -> CoinGeckoClient:
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client
