"""
ApiHealthMonitor — background daemon that periodically verifies every
external dependency the bot relies on, so the dashboard can show at a
glance which APIs are alive RIGHT NOW:

    • Bybit public   (market data: walls, klines — the primary source)
    • Bybit auth     (the user's API key: trading/balance access)
    • Binance        (OI estimation + kline fallback; Render's shared IP
                      gets temp-banned by Binance regularly — this check
                      surfaces the ban and the remaining time)
    • OKX            (order book aggregation leg)
    • Hyperliquid    (order book aggregation leg)
    • Telegram       (alert delivery channel)

Design notes:
    - One light request per service per cycle (default 120 s), run in
      parallel with hard timeouts — the whole sweep costs < 7 s.
    - Results are cached in memory; /api/health/apis just reads cache.
    - Bybit auth uses the same key-resolution path as the trade executor
      (config.bot_settings._resolve_bybit_keys) so the status reflects
      exactly the credentials the bot trades with.
    - Binance ban detection parses the "banned until <ms>" timestamp from
      the -1003 error and reports minutes remaining.
"""

import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

CHECK_INTERVAL_SEC = 120          # one sweep every 2 minutes
REQ_TIMEOUT = 6                   # per-request hard timeout
USER_AGENT = 'svv-bot-health/1.0'

# Status values the frontend understands:
#   ok          green   — responded correctly
#   down        red     — network error / bad response
#   banned      red     — Binance IP ban (detail has minutes left)
#   auth_error  red     — key rejected (invalid/expired/permissions)
#   off         gray    — not configured (no keys / disabled)


class ApiHealthMonitor:

    def __init__(self):
        self._lock = threading.Lock()
        self._results: Dict[str, Dict] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': USER_AGENT})

    # ------------------------------------------------------------
    # Individual checks — each returns a result dict
    # ------------------------------------------------------------

    def _mk(self, key: str, label: str, status: str, detail: str = '',
            latency_ms: Optional[int] = None) -> Dict:
        return {'key': key, 'label': label, 'status': status,
                'detail': detail, 'latency_ms': latency_ms,
                'checked_ts': time.time()}

    def _timed_get(self, url: str, **kw):
        t0 = time.time()
        r = self._session.get(url, timeout=REQ_TIMEOUT, **kw)
        return r, int((time.time() - t0) * 1000)

    def _check_bybit_public(self) -> Dict:
        try:
            r, ms = self._timed_get('https://api.bybit.com/v5/market/time')
            if r.status_code == 200 and r.json().get('retCode') == 0:
                return self._mk('bybit', 'Bybit', 'ok', 'market data', ms)
            return self._mk('bybit', 'Bybit', 'down',
                            f'HTTP {r.status_code}', ms)
        except Exception as e:
            return self._mk('bybit', 'Bybit', 'down', str(e)[:80])

    def _check_bybit_auth(self) -> Dict:
        """Verify the user's trading key with a read-only balance call."""
        try:
            from config.bot_settings import _resolve_bybit_keys
            api_key, api_secret = _resolve_bybit_keys()
        except Exception:
            api_key = api_secret = None
        if not api_key or not api_secret:
            return self._mk('bybit_auth', 'Bybit Key', 'off',
                            'ключі не налаштовані')
        try:
            from pybit.unified_trading import HTTP
            t0 = time.time()
            client = HTTP(api_key=api_key, api_secret=api_secret,
                          timeout=REQ_TIMEOUT)
            resp = client.get_wallet_balance(accountType='UNIFIED')
            ms = int((time.time() - t0) * 1000)
            if resp.get('retCode') == 0:
                return self._mk('bybit_auth', 'Bybit Key', 'ok',
                                'ключ активний, доступ є', ms)
            return self._mk('bybit_auth', 'Bybit Key', 'auth_error',
                            f"retCode {resp.get('retCode')}: "
                            f"{str(resp.get('retMsg'))[:60]}", ms)
        except Exception as e:
            msg = str(e)
            # pybit raises on auth failures too — classify them as
            # auth_error so the UI distinguishes "key dead" from "net down"
            if any(t in msg for t in ('10003', '10004', '33004',
                                       'invalid', 'expired', 'permission')):
                return self._mk('bybit_auth', 'Bybit Key', 'auth_error',
                                msg[:80])
            return self._mk('bybit_auth', 'Bybit Key', 'down', msg[:80])

    def _check_binance(self) -> Dict:
        try:
            r, ms = self._timed_get('https://fapi.binance.com/fapi/v1/ping')
            if r.status_code == 200:
                return self._mk('binance', 'Binance', 'ok',
                                'OI + kline fallback', ms)
            if r.status_code in (418, 429):
                # Parse "banned until 1781130179414" → minutes remaining
                detail = 'IP ban / rate limit'
                m = re.search(r'banned until (\d{13})', r.text or '')
                if m:
                    left = (int(m.group(1)) / 1000.0 - time.time()) / 60.0
                    if left > 0:
                        detail = f'IP бан ще ~{int(left) + 1} хв'
                return self._mk('binance', 'Binance', 'banned', detail, ms)
            return self._mk('binance', 'Binance', 'down',
                            f'HTTP {r.status_code}', ms)
        except Exception as e:
            return self._mk('binance', 'Binance', 'down', str(e)[:80])

    def _check_okx(self) -> Dict:
        try:
            r, ms = self._timed_get(
                'https://www.okx.com/api/v5/public/time')
            if r.status_code == 200 and r.json().get('code') == '0':
                return self._mk('okx', 'OKX', 'ok', 'order book leg', ms)
            return self._mk('okx', 'OKX', 'down',
                            f'HTTP {r.status_code}', ms)
        except Exception as e:
            return self._mk('okx', 'OKX', 'down', str(e)[:80])

    def _check_hyperliquid(self) -> Dict:
        try:
            t0 = time.time()
            r = self._session.post('https://api.hyperliquid.xyz/info',
                                   json={'type': 'meta'},
                                   timeout=REQ_TIMEOUT)
            ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return self._mk('hyperliquid', 'Hyperliquid', 'ok',
                                'order book leg', ms)
            return self._mk('hyperliquid', 'Hyperliquid', 'down',
                            f'HTTP {r.status_code}', ms)
        except Exception as e:
            return self._mk('hyperliquid', 'Hyperliquid', 'down',
                            str(e)[:80])

    def _check_telegram(self) -> Dict:
        token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
        if not token:
            return self._mk('telegram', 'Telegram', 'off',
                            'токен не налаштований')
        try:
            r, ms = self._timed_get(
                f'https://api.telegram.org/bot{token}/getMe')
            j = r.json() if r.status_code in (200, 401) else {}
            if r.status_code == 200 and j.get('ok'):
                name = (j.get('result') or {}).get('username', '')
                return self._mk('telegram', 'Telegram', 'ok',
                                f'@{name}' if name else 'бот активний', ms)
            if r.status_code == 401:
                return self._mk('telegram', 'Telegram', 'auth_error',
                                'токен відхилено (401)', ms)
            return self._mk('telegram', 'Telegram', 'down',
                            f'HTTP {r.status_code}', ms)
        except Exception as e:
            return self._mk('telegram', 'Telegram', 'down', str(e)[:80])

    # ------------------------------------------------------------
    # Sweep + daemon
    # ------------------------------------------------------------

    def run_sweep(self) -> None:
        checks = [self._check_bybit_public, self._check_bybit_auth,
                  self._check_binance, self._check_okx,
                  self._check_hyperliquid, self._check_telegram]
        results = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for res in pool.map(lambda fn: fn(), checks):
                results[res['key']] = res
        with self._lock:
            self._results = results
        bad = [r['label'] for r in results.values()
               if r['status'] in ('down', 'banned', 'auth_error')]
        if bad:
            print(f"[APIHEALTH] sweep: issues with {', '.join(bad)}")

    def _loop(self):
        while not self._stop:
            try:
                self.run_sweep()
            except Exception as e:
                print(f'[APIHEALTH] sweep error: {e}')
            for _ in range(CHECK_INTERVAL_SEC):
                if self._stop:
                    return
                time.sleep(1)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name='api-health')
        self._thread.start()
        print('[APIHEALTH] Monitor started '
              f'(interval {CHECK_INTERVAL_SEC}s)')

    def get_status(self) -> Dict:
        with self._lock:
            results = list(self._results.values())
        order = ['bybit', 'bybit_auth', 'binance', 'okx',
                 'hyperliquid', 'telegram']
        results.sort(key=lambda r: order.index(r['key'])
                     if r['key'] in order else 99)
        return {
            'running': bool(self._thread and self._thread.is_alive()),
            'interval_sec': CHECK_INTERVAL_SEC,
            'services': results,
        }


_monitor: Optional[ApiHealthMonitor] = None
_monitor_lock = threading.Lock()


def get_api_health_monitor() -> Optional[ApiHealthMonitor]:
    return _monitor


def init_api_health_monitor() -> ApiHealthMonitor:
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = ApiHealthMonitor()
        return _monitor
