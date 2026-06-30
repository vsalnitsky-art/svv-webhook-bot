"""
Funding Rate Monitor v3.0 — Smart Watchlist

Entry: funding ≤ -1%
Priority: funding getting MORE negative AND price rising → Telegram alert
Watch: 5 days, then auto-delete
Manual delete via API
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

SCAN_INTERVAL = 60          # 1 minute
ENTRY_THRESHOLD = -1.0
WATCH_DAYS = 5
DB_KEY = 'funding_watchlist'
DB_KEY_THRESHOLD = 'funding_entry_threshold'  # user-tunable entry threshold (%)
DB_KEY_MIN_VOLUME = 'funding_min_volume_usd'  # user-tunable min 24h turnover (USD)
MIN_VOLUME_USD = 0.0          # default: 0 = volume filter off
TREND_WINDOW = 30           # 30 scans × 1min = 30 min trend


class FundingMonitor:
    def __init__(self, bybit_connector=None, db=None, notifier=None,
                 scan_interval: int = SCAN_INTERVAL):
        self.bybit = bybit_connector
        self.db = db
        self.notifier = notifier
        self.scan_interval = scan_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._watchlist: Dict[str, Dict] = {}
        self._scan_count: int = 0
        self._total_coins: int = 0
        self._errors: int = 0
        self._alerted: set = set()
        # Entry threshold (%, negative). User-tunable from the UI header and
        # persisted in DB so it survives restarts. Falls back to the module
        # default ENTRY_THRESHOLD on first run / read error.
        self._entry_threshold: float = ENTRY_THRESHOLD
        self._load_threshold()
        # Minimum 24h turnover (USD) for a coin to enter the watchlist. Lets the
        # user filter out illiquid coins where extreme funding is noise rather
        # than a tradeable squeeze. 0 = filter off. Tunable from the UI header.
        self._min_volume: float = MIN_VOLUME_USD
        self._load_min_volume()
        # Guards against overlapping scans (daemon cycle vs. manual rescan
        # triggered by a UI filter change). Non-blocking: if one is running,
        # the other skips this round.
        self._scan_lock = threading.Lock()
        self._load_watchlist()

    def _load_threshold(self):
        """Restore the user-set entry threshold from DB (default ENTRY_THRESHOLD)."""
        if not self.db:
            return
        try:
            raw = self.db.get_setting(DB_KEY_THRESHOLD, None)
            if raw is not None:
                self._entry_threshold = float(raw)
        except Exception as e:
            print(f"[FUNDING] threshold load error: {e}")

    def set_entry_threshold(self, value: float) -> float:
        """Set and persist the entry threshold (%). Clamped to a sane negative
        range so the scanner stays meaningful (funding entries are negative).
        Returns the value actually applied."""
        try:
            v = float(value)
        except (ValueError, TypeError):
            return self._entry_threshold
        # Keep it negative and within a realistic band: 0 down to -5%.
        v = max(-5.0, min(-0.001, v))
        v = round(v, 4)
        with self._lock:
            self._entry_threshold = v
        if self.db:
            try:
                self.db.set_setting(DB_KEY_THRESHOLD, v)
            except Exception as e:
                print(f"[FUNDING] threshold save error: {e}")
        print(f"[FUNDING] entry threshold set to {v}%")
        return v

    def _load_min_volume(self):
        """Restore the user-set min 24h volume from DB (default MIN_VOLUME_USD)."""
        if not self.db:
            return
        try:
            raw = self.db.get_setting(DB_KEY_MIN_VOLUME, None)
            if raw is not None:
                self._min_volume = max(0.0, float(raw))
        except Exception as e:
            print(f"[FUNDING] min volume load error: {e}")

    def set_min_volume(self, value) -> float:
        """Set and persist the minimum 24h turnover (USD) for watchlist entry.
        0 disables the filter. Returns the value actually applied."""
        try:
            v = max(0.0, float(value))
        except (ValueError, TypeError):
            return self._min_volume
        with self._lock:
            self._min_volume = v
        if self.db:
            try:
                self.db.set_setting(DB_KEY_MIN_VOLUME, v)
            except Exception as e:
                print(f"[FUNDING] min volume save error: {e}")
        print(f"[FUNDING] min 24h volume set to ${v:,.0f}")
        return v

    def get_symbols(self) -> List[str]:
        """Lightweight list of currently tracked symbols (for cross-module use,
        e.g. feeding the Fuel Auto-Filter). Cheap — no per-coin computation."""
        with self._lock:
            return list(self._watchlist.keys())

    def get_rates(self) -> Dict[str, float]:
        """{symbol: current funding rate (%)} for tracked coins — latest sample,
        falling back to the trigger rate. Used by the FF table to show the live
        funding of funding-sourced coins."""
        out = {}
        with self._lock:
            for sym, c in self._watchlist.items():
                rs = c.get('rates') or []
                if rs:
                    out[sym] = rs[-1].get('r')
                elif c.get('trigger_rate') is not None:
                    out[sym] = c.get('trigger_rate')
        return out

    def get_volumes(self) -> Dict[str, float]:
        """{symbol: 24h quote volume (USD)} for tracked coins."""
        with self._lock:
            return {sym: float(c.get('volume') or 0)
                    for sym, c in self._watchlist.items()}

    def get_next_funding(self) -> Dict[str, int]:
        """{symbol: nextFundingTime (ms)} for tracked coins — for the
        'time left until funding' countdown."""
        with self._lock:
            return {sym: int(c.get('next_funding') or 0)
                    for sym, c in self._watchlist.items()
                    if c.get('next_funding')}

    def trigger_rescan(self):
        """Run a scan immediately (background thread) so UI filter changes take
        effect within seconds instead of waiting for the next cycle."""
        def _run():
            try:
                self._scan()
            except Exception as e:
                print(f"[FUNDING] manual rescan error: {e}")
        threading.Thread(target=_run, daemon=True, name="FundingRescan").start()

    def _load_watchlist(self):
        if not self.db:
            return
        try:
            saved = self.db.get_setting(DB_KEY, {})
            if isinstance(saved, dict):
                self._watchlist = saved
                for sym, data in self._watchlist.items():
                    if data.get('alerted'):
                        self._alerted.add(sym)
                if self._watchlist:
                    print(f"[FUNDING] Restored {len(self._watchlist)} tracked coins")
        except Exception as e:
            print(f"[FUNDING] Load error: {e}")

    def _save_watchlist(self):
        if not self.db:
            return
        try:
            self.db.set_setting(DB_KEY, self._watchlist)
        except Exception as e:
            if self._scan_count <= 3:
                print(f"[FUNDING] Save error: {e}")

    def start(self):
        if self._running:
            return
        # Check DB toggle
        if self.db and self.db.get_setting('funding_enabled', '1') != '1':
            print("[FUNDING] Disabled in DB, not starting")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FundingMonitor")
        self._thread.start()
        print(f"[FUNDING] Started: every {self.scan_interval}s, entry: {self._entry_threshold}%, "
              f"watch: {WATCH_DAYS}d, tracked: {len(self._watchlist)}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def is_enabled(self) -> bool:
        if not self.db:
            return self._running
        return self.db.get_setting('funding_enabled', '1') == '1'
    
    def set_enabled(self, enabled: bool) -> bool:
        """Turn module on/off and persist state."""
        if self.db:
            self.db.set_setting('funding_enabled', '1' if enabled else '0')
        if enabled and not self._running:
            # Reset the DB check we added above so start() proceeds
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True, name="FundingMonitor")
            self._thread.start()
            print("[FUNDING] ✅ Enabled and started")
        elif not enabled and self._running:
            self._running = False
            print("[FUNDING] ⏸️ Disabled")
        return True

    def _loop(self):
        print("[FUNDING] Scan thread started")
        try:
            self._scan()
            while self._running:
                time.sleep(self.scan_interval)
                if self._running:
                    self._scan()
        except Exception as e:
            print(f"[FUNDING] Thread crashed: {e}")
            import traceback
            traceback.print_exc()

    def _scan(self):
        if not self.bybit:
            return
        # Skip if a scan (daemon or manual rescan) is already running.
        if not self._scan_lock.acquire(blocking=False):
            return
        try:
            self._scan_body()
        finally:
            self._scan_lock.release()

    def _scan_body(self):
        try:
            tickers = self.bybit.get_tickers(category='linear')
            if not tickers:
                return
            now = datetime.now(timezone.utc)
            now_str = now.strftime('%Y-%m-%d %H:%M')
            self._scan_count += 1

            rates: Dict[str, Dict] = {}
            for t in tickers:
                symbol = t.get('symbol', '')
                if not symbol.endswith('USDT'):
                    continue
                try:
                    rate = float(t.get('fundingRate', '0'))
                    price = float(t.get('lastPrice', '0'))
                    # turnover24h = 24h quote volume in USDT (USD). Best
                    # "large volume" proxy on Bybit v5 linear tickers.
                    volume = float(t.get('turnover24h', '0') or 0)
                    # nextFundingTime = ms timestamp of the next funding
                    # settlement → lets us show "time left until funding".
                    next_funding = int(float(t.get('nextFundingTime') or 0))
                except (ValueError, TypeError):
                    continue
                rates[symbol] = {'rate': rate, 'price': price, 'volume': volume,
                                 'next_funding': next_funding}

            self._total_coins = len(rates)
            threshold = self._entry_threshold / 100
            min_vol = self._min_volume
            new_added = 0
            expired_count = 0

            with self._lock:
                for symbol, data in rates.items():
                    # Entry requires BOTH: funding ≤ threshold AND (when the
                    # volume filter is on) 24h turnover ≥ min volume.
                    if (symbol not in self._watchlist
                            and data['rate'] <= threshold
                            and (min_vol <= 0 or data['volume'] >= min_vol)):
                        self._watchlist[symbol] = {
                            'first_seen': now_str,
                            'trigger_rate': round(data['rate'] * 100, 4),
                            'price_at_trigger': data['price'],
                            'volume': data['volume'],
                            'next_funding': data.get('next_funding', 0),
                            'alerted': False,
                            'manual': False,
                            'rates': [],
                        }
                        new_added += 1

                for symbol in list(self._watchlist.keys()):
                    coin = self._watchlist[symbol]
                    if symbol in rates:
                        r = rates[symbol]
                        coin['volume'] = r['volume']   # keep live 24h volume
                        coin['next_funding'] = r.get('next_funding', 0)
                        coin['rates'].append({
                            't': now_str,
                            'r': round(r['rate'] * 100, 4),
                            'p': r['price'],
                        })
                        if len(coin['rates']) > 7200:
                            coin['rates'] = coin['rates'][-7200:]

                for symbol, coin in self._watchlist.items():
                    if coin.get('alerted'):
                        continue
                    if self._check_priority(coin):
                        coin['alerted'] = True
                        self._alerted.add(symbol)
                        self._send_alert(symbol, coin)

                cutoff = (now - timedelta(days=WATCH_DAYS)).strftime('%Y-%m-%d %H:%M')
                expired = [s for s, c in self._watchlist.items()
                           if c.get('first_seen', '') < cutoff]
                for s in expired:
                    del self._watchlist[s]
                    self._alerted.discard(s)
                expired_count = len(expired)

                # Volume prune: drop AUTO-tracked coins that no longer meet the
                # min-volume filter, so raising "Vol ≥" immediately trims the
                # list to liquid coins. Manually-added coins are kept regardless.
                if min_vol > 0:
                    low_vol = [s for s, c in self._watchlist.items()
                               if not c.get('manual')
                               and c.get('volume', 0) < min_vol]
                    for s in low_vol:
                        del self._watchlist[s]
                        self._alerted.discard(s)
                    expired_count += len(low_vol)

                # Funding-threshold prune: drop AUTO-tracked coins whose CURRENT
                # funding no longer meets the threshold. The filter is LIVE — a
                # coin stays only while its funding is ≤ "Entry ≤ X%" right now;
                # once funding recovers above the threshold it leaves. `threshold`
                # is decimal (e.g. -0.01); current funding comes from this tick's
                # ticker, falling back to the last recorded rate (% → decimal).
                # Manually-added coins (✋) are never pruned.
                off_thr = []
                for s, c in self._watchlist.items():
                    if c.get('manual'):
                        continue
                    if s in rates:
                        cur = rates[s]['rate']            # decimal, fresh
                    else:
                        rs = c.get('rates') or []
                        cur = (rs[-1]['r'] / 100.0) if rs else None
                    if cur is not None and cur > threshold:
                        off_thr.append(s)
                for s in off_thr:
                    del self._watchlist[s]
                    self._alerted.discard(s)
                expired_count += len(off_thr)

            self._save_watchlist()

            if self._scan_count <= 2 or self._scan_count % 12 == 0 or new_added > 0:
                print(f"[FUNDING] #{self._scan_count}: {self._total_coins} coins, "
                      f"{len(self._watchlist)} tracked"
                      f"{f', +{new_added} new' if new_added else ''}"
                      f"{f', -{expired_count} expired' if expired_count else ''}")

        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 10 == 0:
                print(f"[FUNDING] Error #{self._errors}: {e}")

    def _check_priority(self, coin: Dict) -> bool:
        """PRIORITY = funding MORE negative AND price rising over last 30min."""
        rates = coin.get('rates', [])
        if len(rates) < TREND_WINDOW + 1:
            return False
        recent = rates[-(TREND_WINDOW + 1):]
        if recent[-1]['r'] > -1.5:
            return False
        funding_down = sum(1 for i in range(1, len(recent)) if recent[i]['r'] < recent[i-1]['r'])
        price_up = sum(1 for i in range(1, len(recent)) if recent[i]['p'] > recent[i-1]['p'])
        min_signals = max(3, TREND_WINDOW * 2 // 3)
        return funding_down >= min_signals and price_up >= min_signals

    def _send_alert(self, symbol: str, coin: Dict):
        if not self.notifier:
            print(f"[FUNDING] PRIORITY (no TG): {symbol}")
            return
        try:
            rates = coin.get('rates', [])
            cur = rates[-1] if rates else {}
            trigger = coin.get('trigger_rate', 0)
            cur_rate = cur.get('r', trigger)
            cur_price = cur.get('p', 0)
            trig_price = coin.get('price_at_trigger', 0)
            pchg = ((cur_price - trig_price) / trig_price * 100) if trig_price else 0

            msg = (
                f"\U0001f6a8 <b>FUNDING ALERT: {symbol}</b>\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"\U0001f4ca Funding: {trigger:+.3f}% \u2192 <b>{cur_rate:+.3f}%</b>\n"
                f"\U0001f4b0 Price: ${trig_price:,.6g} \u2192 <b>${cur_price:,.6g}</b> ({pchg:+.2f}%)\n"
                f"\u26a1 <b>Funding falling + Price rising = Short Squeeze risk</b>\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"\U0001f550 Since: {coin.get('first_seen', '')}"
            )
            self.notifier.send_message(msg)
            print(f"[FUNDING] TG alert: {symbol} ({cur_rate:+.3f}%, price {pchg:+.2f}%)")
        except Exception as e:
            print(f"[FUNDING] Alert error {symbol}: {e}")

    def get_watchlist(self) -> Dict:
        with self._lock:
            coins = []
            now = datetime.now(timezone.utc)
            for symbol, data in self._watchlist.items():
                rates = data.get('rates', [])
                cur_rate = rates[-1]['r'] if rates else data.get('trigger_rate', 0)
                cur_price = rates[-1]['p'] if rates else data.get('price_at_trigger', 0)
                rvals = [r['r'] for r in rates] if rates else [cur_rate]

                first_seen = data.get('first_seen', '')
                try:
                    fs = datetime.strptime(first_seen, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                    hours_tracked = (now - fs).total_seconds() / 3600
                    hours_left = max(0, (fs + timedelta(days=WATCH_DAYS) - now).total_seconds() / 3600)
                except:
                    hours_tracked = 0
                    hours_left = WATCH_DAYS * 24

                trig_price = data.get('price_at_trigger', 0)
                pchg = round((cur_price - trig_price) / trig_price * 100, 2) if trig_price else 0

                is_priority = data.get('alerted', False) or self._check_priority(data)
                ft = self._calc_trend(rates, 'r') if len(rates) >= 3 else 0
                pt = self._calc_trend(rates, 'p') if len(rates) >= 3 else 0

                coins.append({
                    'symbol': symbol,
                    'trigger_rate': data.get('trigger_rate', 0),
                    'current_rate': cur_rate,
                    'min_rate': min(rvals),
                    'max_rate': max(rvals),
                    'current_price': cur_price,
                    'price_change': pchg,
                    'hours_tracked': round(hours_tracked, 1),
                    'hours_left': round(hours_left, 1),
                    'data_points': len(rates),
                    'first_seen': first_seen,
                    'is_priority': is_priority,
                    'alerted': data.get('alerted', False),
                    'manual': data.get('manual', False),
                    'funding_trend': ft,
                    'price_trend': pt,
                    'volume': data.get('volume', 0),
                    'next_funding': data.get('next_funding', 0),
                })

            coins.sort(key=lambda x: (not x['is_priority'], x['current_rate']))
            return {
                'coins': coins,
                'total_tracked': len(coins),
                'scan_count': self._scan_count,
                'total_coins': self._total_coins,
                'errors': self._errors,
                'running': self._running,
                'threshold': abs(self._entry_threshold),
                'threshold_raw': self._entry_threshold,
                'min_volume': self._min_volume,
                'watch_days': WATCH_DAYS,
            }

    def _calc_trend(self, rates: List[Dict], field: str) -> int:
        if len(rates) < 3:
            return 0
        recent = rates[-min(TREND_WINDOW, len(rates)):]
        ups = sum(1 for i in range(1, len(recent)) if recent[i][field] > recent[i-1][field])
        downs = sum(1 for i in range(1, len(recent)) if recent[i][field] < recent[i-1][field])
        if ups > downs + 1:
            return 1
        if downs > ups + 1:
            return -1
        return 0

    def add_coin(self, symbol: str) -> Dict:
        """Manually add a coin to watchlist. Fetches current rate from Bybit."""
        symbol = symbol.upper().replace('.P', '')
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        with self._lock:
            if symbol in self._watchlist:
                return {'ok': False, 'reason': f'{symbol} already tracked'}
        
        # Fetch current rate
        if not self.bybit:
            return {'ok': False, 'reason': 'Bybit not connected'}
        
        try:
            tickers = self.bybit.get_tickers(category='linear')
            found = None
            for t in tickers:
                if t.get('symbol') == symbol:
                    found = t
                    break
            
            if not found:
                return {'ok': False, 'reason': f'{symbol} not found on Bybit'}
            
            rate = float(found.get('fundingRate', '0'))
            price = float(found.get('lastPrice', '0'))
            volume = float(found.get('turnover24h', '0') or 0)
            now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')

            with self._lock:
                self._watchlist[symbol] = {
                    'first_seen': now_str,
                    'trigger_rate': round(rate * 100, 4),
                    'price_at_trigger': price,
                    'volume': volume,
                    'alerted': False,
                    'manual': True,
                    'rates': [{
                        't': now_str,
                        'r': round(rate * 100, 4),
                        'p': price,
                    }],
                }
            
            self._save_watchlist()
            print(f"[FUNDING] ➕ Manually added: {symbol} (rate {rate*100:.3f}%, price ${price:,.6g})")
            return {'ok': True, 'symbol': symbol, 'rate': round(rate * 100, 4), 'price': price}
        
        except Exception as e:
            return {'ok': False, 'reason': str(e)}
    
    def get_coin_rates(self, symbol: str) -> Dict:
        """Full rate history for a single coin (for chart)."""
        with self._lock:
            if symbol not in self._watchlist:
                return {'symbol': symbol, 'found': False, 'rates': []}
            coin = self._watchlist[symbol]
            return {
                'symbol': symbol,
                'found': True,
                'trigger_rate': coin.get('trigger_rate', 0),
                'price_at_trigger': coin.get('price_at_trigger', 0),
                'first_seen': coin.get('first_seen', ''),
                'rates': coin.get('rates', []),
            }
    
    def remove_coin(self, symbol: str) -> bool:
        with self._lock:
            if symbol in self._watchlist:
                del self._watchlist[symbol]
                self._alerted.discard(symbol)
                self._save_watchlist()
                print(f"[FUNDING] Removed: {symbol}")
                return True
        return False


_instance: Optional[FundingMonitor] = None

def get_funding_monitor() -> Optional[FundingMonitor]:
    return _instance

def init_funding_monitor(bybit_connector=None, db=None, notifier=None) -> FundingMonitor:
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = FundingMonitor(bybit_connector=bybit_connector, db=db, notifier=notifier)
    return _instance
