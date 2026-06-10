"""
liqmap_signal_scanner.py — watchlist-wide Liquidity Map signal engine.

Every cycle (default 120s) it walks the SMC watchlist and computes a
composite 0–100 entry score per direction from the same five inputs the
dashboard Liquidity Map shows:

    Component        Weight  LONG-favorable when
    ─────────────────────────────────────────────────────────────
    Forecast 1H/4H     30    both TFs point LONG with confidence
    Book imbalance     20    bids outweigh asks (3-exchange USD)
    Liq fuel ratio     20    short-liq USD above >> long-liq below
    Squeeze            20    deep compression (energy loaded) — the
                             squeeze itself is direction-neutral, it
                             multiplies conviction of the other inputs
    Manipulation       10    LOW spoof % (book is trustworthy)

(SHORT score is the mirror.) When liq-zone data is missing for a symbol
(daemon only force-tracks BTC/ETH + on-demand), that weight is
redistributed proportionally over the remaining components rather than
silently scoring zero — otherwise alt-coin signals would be capped at 80.

A Telegram alert fires when score ≥ threshold (default 75), with a
per-(symbol, direction) cooldown (default 60 min) and a "must drop below
threshold-10 before re-arming" hysteresis so a score hovering at the
threshold doesn't machine-gun alerts at every cooldown expiry.

Settings (DB):
    liqmap_signal_enabled    bool, default False  (operator opts in)
    liqmap_signal_threshold  int 50..95, default 75
    liqmap_signal_cooldown   minutes, default 60

The scanner is deliberately gentle on rate limits: symbols are processed
sequentially with a small delay; one cycle over a 30-symbol watchlist
costs ~30 aggregated book fetches + ~30 kline fetches spread over ~45s.
"""

import time
import threading
from typing import Dict, List, Optional

CYCLE_SEC = 120
PER_SYMBOL_DELAY = 1.2
# Manipulation heartbeat: light tracker-feed pass between scoring cycles.
# HEARTBEAT_SEC × SCORE_EVERY_TICKS ≈ CYCLE_SEC keeps the scoring cadence.
HEARTBEAT_SEC = 30
SCORE_EVERY_TICKS = 4
HEARTBEAT_SYMBOL_DELAY = 0.3
DEFAULT_THRESHOLD = 75
DEFAULT_COOLDOWN_MIN = 60
REARM_DROP = 10          # score must dip below (threshold - this) to re-arm
LIQ_BAND_PCT = 0.05      # liq fuel counted within ±5% of price


class LiqmapSignalScanner:
    def __init__(self, db=None, notifier=None):
        self.db = db
        self.notifier = notifier
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._lock = threading.Lock()
        # (symbol, side) -> {'last_alert_ts': float, 'armed': bool}
        self._alert_state: Dict[tuple, Dict] = {}
        self._last_scores: Dict[str, Dict] = {}   # symbol -> latest breakdown
        self._cycle_count = 0
    
    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name='LiqmapSignalScanner')
        self._thread.start()
        print('[LIQSIG] Scanner started')
    
    def stop(self):
        self._stop = True
    
    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    
    def _settings(self) -> Dict:
        try:
            enabled = str(self.db.get_setting('liqmap_signal_enabled', 'false')).lower() in ('true', '1')
            threshold = int(float(self.db.get_setting('liqmap_signal_threshold', DEFAULT_THRESHOLD)))
            cooldown = int(float(self.db.get_setting('liqmap_signal_cooldown', DEFAULT_COOLDOWN_MIN)))
        except Exception:
            enabled, threshold, cooldown = False, DEFAULT_THRESHOLD, DEFAULT_COOLDOWN_MIN
        threshold = max(50, min(95, threshold))
        cooldown = max(5, min(720, cooldown))
        return {'enabled': enabled, 'threshold': threshold, 'cooldown_min': cooldown}
    
    def _watchlist(self) -> List[str]:
        try:
            import json
            raw = self.db.get_setting('smc_watchlist', '[]')
            wl = json.loads(raw) if isinstance(raw, str) else (raw or [])
            return [str(s).upper() for s in wl if s]
        except Exception as e:
            print(f'[LIQSIG] watchlist read error: {e}')
            return []
    
    # ------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------
    
    def compute_score(self, symbol: str) -> Optional[Dict]:
        """Compute LONG and SHORT composite scores with full breakdown.
        Returns None when core data (order book) is unavailable."""
        # --- Order book (walls + imbalance + manipulation feed) ---
        try:
            from detection.orderbook_collector import (
                fetch_aggregated_orderbook, compute_walls_buckets_v3)
            snap = fetch_aggregated_orderbook(symbol)
            if snap is None:
                return None
            walls = compute_walls_buckets_v3(snap, top_n=8)
            if not walls:
                return None
        except Exception as e:
            print(f'[LIQSIG] {symbol} book error: {e}')
            return None
        
        # Feed the manipulation tracker so its window keeps filling even
        # when nobody is watching the dashboard — the scanner becomes the
        # heartbeat for watchlist symbols.
        manip = None
        try:
            from detection.manipulation_tracker import get_manipulation_tracker
            mt = get_manipulation_tracker()
            mt.update(symbol, walls)
            manip = mt.get_state(symbol)
        except Exception:
            pass
        
        # --- Forecast (cache-first; SMC keeps watchlist fresh) ---
        f1 = f4 = None
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe is not None:
                cached = fe.get(symbol)
                if cached:
                    f1 = cached.get('forecast_1h')
                    f4 = cached.get('forecast_4h')
        except Exception:
            pass
        
        # --- Squeeze (15m — the dashboard's default lens) ---
        squeeze = None
        try:
            from detection.market_data import get_market_data
            from detection.squeeze import calc_squeeze
            md = get_market_data()
            if md is not None:
                kl = md.fetch_klines(symbol, interval='15m', limit=120)
                if kl:
                    squeeze = calc_squeeze(kl)
        except Exception:
            pass
        
        # --- Liquidation zones (only if daemon has data) ---
        liq_above_usd = liq_below_usd = None
        try:
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm is not None:
                state = lm.get_state(symbol=symbol, lookback_hours=24)
                mark = state.get('mark_price')
                zones = state.get('cluster_zones') or []
                if mark and zones:
                    above = below = 0.0
                    for z in zones:
                        zc = (z['price_low'] + z['price_high']) / 2
                        if abs(zc - mark) / mark > LIQ_BAND_PCT:
                            continue
                        if zc > mark:
                            above += z.get('total_usd', 0)
                        else:
                            below += z.get('total_usd', 0)
                    if above + below > 0:
                        liq_above_usd, liq_below_usd = above, below
        except Exception:
            pass
        
        # =========================================================
        # Component sub-scores in [-1, +1] (positive favors LONG)
        # =========================================================
        # Forecast: blended side×confidence as on the dashboard
        s1 = ((f1 or {}).get('side') or 0) * ((f1 or {}).get('confidence') or 0)
        s4 = ((f4 or {}).get('side') or 0) * ((f4 or {}).get('confidence') or 0)
        forecast_sub = max(-1.0, min(1.0, (s1 * 0.4 + s4 * 0.6) / 100.0))
        
        # Imbalance: ±30% book imbalance saturates the sub-score
        imb = walls.get('imbalance_pct') or 0
        imbalance_sub = max(-1.0, min(1.0, imb / 30.0))
        
        # Liq fuel: log-ish ratio of fuel above vs below; 3× ratio saturates
        liq_sub = None
        if liq_above_usd is not None:
            num = liq_above_usd - liq_below_usd
            den = liq_above_usd + liq_below_usd
            liq_sub = max(-1.0, min(1.0, (num / den) * 1.5)) if den > 0 else 0.0
        
        # Squeeze: direction-neutral energy in [0, 1]
        squeeze_energy = 0.0
        if squeeze and squeeze.get('ok') and squeeze.get('squeeze_on'):
            squeeze_energy = (squeeze.get('probability') or 0) / 100.0
        
        # Manipulation trust in [0, 1]: 0% spoofs → 1.0, ≥80% → 0
        trust = 1.0
        if manip and not manip.get('warming_up'):
            trust = max(0.0, 1.0 - (manip.get('pct', 0) / 80.0))
        
        # =========================================================
        # Directional composites
        # =========================================================
        # Weights; liq weight redistributed when zone data is absent
        w_forecast, w_imb, w_liq, w_squeeze, w_manip = 30, 20, 20, 20, 10
        if liq_sub is None:
            # Redistribute the 20 liq points proportionally across the
            # remaining directional components (30:20:20 → 37.5:25:25)
            w_forecast, w_imb, w_squeeze = 37.5, 25.0, 25.0
            w_liq = 0.0
        
        def directional(sign: int) -> float:
            d_forecast = max(0.0, forecast_sub * sign)        # 0..1
            d_imb = max(0.0, imbalance_sub * sign)
            d_liq = max(0.0, (liq_sub or 0.0) * sign)
            score = (w_forecast * d_forecast
                     + w_imb * d_imb
                     + w_liq * d_liq
                     + w_squeeze * squeeze_energy)
            # Manipulation scales the whole conviction, and its weight is
            # granted only at full trust
            score = score * (0.5 + 0.5 * trust) + w_manip * trust * (
                1 if (d_forecast > 0 or d_imb > 0) else 0)
            return max(0.0, min(100.0, score))
        
        breakdown = {
            'symbol': symbol,
            'ts': time.time(),
            'mid_price': walls.get('mid_price'),
            'long_score': round(directional(+1), 1),
            'short_score': round(directional(-1), 1),
            'forecast_sub': round(forecast_sub, 3),
            'imbalance_pct': round(imb, 2),
            'liq_above_usd': liq_above_usd,
            'liq_below_usd': liq_below_usd,
            'squeeze_on': bool(squeeze and squeeze.get('squeeze_on')),
            'squeeze_prob': (squeeze or {}).get('probability', 0),
            'squeeze_bars': (squeeze or {}).get('bars_in_squeeze', 0),
            'manip_pct': (manip or {}).get('pct') if manip and not manip.get('warming_up') else None,
            'sources': walls.get('sources') or [],
            'f1': f1, 'f4': f4,
        }
        with self._lock:
            self._last_scores[symbol] = breakdown
        return breakdown
    
    # ------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------
    
    def _maybe_alert(self, br: Dict, settings: Dict):
        threshold = settings['threshold']
        cooldown_s = settings['cooldown_min'] * 60
        now = time.time()
        
        for side, score in (('LONG', br['long_score']),
                             ('SHORT', br['short_score'])):
            key = (br['symbol'], side)
            st = self._alert_state.setdefault(key, {'last_alert_ts': 0,
                                                     'armed': True})
            # Hysteresis re-arm
            if not st['armed'] and score < threshold - REARM_DROP:
                st['armed'] = True
            if score < threshold:
                continue
            if not st['armed']:
                continue
            if now - st['last_alert_ts'] < cooldown_s:
                continue
            
            st['last_alert_ts'] = now
            st['armed'] = False
            # Cross-process dedupe: with multiple Gunicorn workers a
            # scanner thread can run in each — per-process cooldown state
            # alone would double-send every alert. The DB check is the
            # shared gate; write-after-send keeps it one small JSON.
            if self._db_cooldown_passed(key, cooldown_s, now):
                self._send_alert(br, side, score, threshold)
                self._db_mark_alert(key, now)
    
    def _db_cooldown_passed(self, key, cooldown_s: float, now: float) -> bool:
        try:
            import json
            raw = self.db.get_setting('liqmap_alert_log', '{}')
            log = json.loads(raw) if raw else {}
            k = f"{key[0]}|{key[1]}"
            return (now - log.get(k, 0)) >= cooldown_s
        except Exception:
            return True  # fail-open: better a rare duplicate than silence
    
    def _db_mark_alert(self, key, now: float):
        try:
            import json
            raw = self.db.get_setting('liqmap_alert_log', '{}')
            log = json.loads(raw) if raw else {}
            log[f"{key[0]}|{key[1]}"] = now
            # Prune entries older than 24h to keep the blob tiny
            cutoff = now - 86400
            log = {k: v for k, v in log.items() if v >= cutoff}
            self.db.set_setting('liqmap_alert_log', json.dumps(log))
        except Exception as e:
            print(f"[LIQSIG] alert log write error: {e}")
    
    def _send_alert(self, br: Dict, side: str, score: float, threshold: int):
        sym = br['symbol']
        emoji = '🟢' if side == 'LONG' else '🔴'
        price = br.get('mid_price')
        price_s = f"${price:,.6g}" if price else '—'
        
        lines = [
            f"💧 <b>LIQUIDITY MAP SIGNAL</b>",
            f"{emoji} <b>{side} {sym}</b> @ {price_s} · score <b>{score:.0f}</b>/100 (поріг {threshold})",
        ]
        f1, f4 = br.get('f1') or {}, br.get('f4') or {}
        def fdesc(f):
            s = f.get('side') or 0
            if not s:
                return '—'
            return f"{'LONG' if s > 0 else 'SHORT'} {f.get('confidence', 0)}%"
        lines.append(f"🔮 Forecast: 1H {fdesc(f1)} · 4H {fdesc(f4)}")
        if br.get('squeeze_on'):
            lines.append(f"⚡ Squeeze: ON {br['squeeze_prob']:.0f}% "
                         f"({br['squeeze_bars']} bars 15m)")
        lines.append(f"⚖️ Imbalance: {br['imbalance_pct']:+.1f}% "
                     f"({'+'.join(br.get('sources') or []) or 'book'})")
        if br.get('liq_above_usd') is not None:
            a, b = br['liq_above_usd'], br['liq_below_usd']
            def fu(v):
                return f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K"
            lines.append(f"🧲 Liq fuel ±5%: {fu(a)} above / {fu(b)} below")
        if br.get('manip_pct') is not None:
            lines.append(f"🎭 Manipulation: {br['manip_pct']:.0f}%")
        else:
            lines.append("🎭 Manipulation: — (статистика накопичується)")
        lines.append(f"\n<i>Сигнал карти ліквідності — підтверди тайминг "
                     f"через SMC структуру перед входом.</i>")
        
        msg = '\n'.join(lines)
        print(f"[LIQSIG] 🔔 {side} {sym} score={score:.0f}")
        if self.notifier:
            try:
                self.notifier.send_message(msg)
            except Exception as e:
                print(f"[LIQSIG] telegram error: {e}")
    
    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    
    def _loop(self):
        # Tick architecture (2026-06-10): the manipulation tracker needs
        # dense sampling to observe wall lifecycles — 120s gaps made
        # classification meaningless (and a promotion bug made it always
        # 0%). Now: every HEARTBEAT_SEC=30 we do a light pass over the
        # watchlist feeding the tracker (book fetch is cached 2.5s, walls
        # computation is microseconds); every SCORE_EVERY_TICKS-th tick we
        # additionally compute full scores + alerts. compute_score's own
        # book fetch hits the aggregator cache from the heartbeat pass, so
        # the scoring tick costs no extra exchange calls.
        tick = 0
        while not self._stop:
            try:
                settings = self._settings()
                if settings['enabled']:
                    symbols = self._watchlist()
                    scoring_tick = (tick % SCORE_EVERY_TICKS == 0)
                    if scoring_tick:
                        self._cycle_count += 1
                    for sym in symbols:
                        if self._stop:
                            break
                        try:
                            if scoring_tick:
                                br = self.compute_score(sym)
                                if br:
                                    self._maybe_alert(br, settings)
                                time.sleep(PER_SYMBOL_DELAY)
                            else:
                                # Heartbeat: feed manipulation tracker only
                                self._manip_heartbeat(sym)
                                time.sleep(HEARTBEAT_SYMBOL_DELAY)
                        except Exception as e:
                            print(f"[LIQSIG] {sym} tick error: {e}")
                    if scoring_tick:
                        # Persist the score snapshot: with multiple Gunicorn
                        # workers the scanner lives in one process while
                        # /status requests may hit another; and worker
                        # restarts wiped scores entirely (watchlist badges
                        # went blank). DB is the shared source of truth.
                        self._persist_scores()
                    if scoring_tick and self._cycle_count % 10 == 1:
                        print(f"[LIQSIG] scoring cycle #{self._cycle_count} "
                              f"done ({len(symbols)} symbols)")
            except Exception as e:
                print(f"[LIQSIG] loop error: {e}")
            
            tick += 1
            # Sleep in 1s chunks so stop() is responsive
            for _ in range(HEARTBEAT_SEC):
                if self._stop:
                    break
                time.sleep(1)
    
    def _manip_heartbeat(self, symbol: str):
        """Light pass: fetch aggregated book, compute walls, feed the
        manipulation tracker. No scoring, no alerts."""
        try:
            from detection.orderbook_collector import (
                fetch_aggregated_orderbook, compute_walls_buckets_v3)
            from detection.manipulation_tracker import get_manipulation_tracker
            snap = fetch_aggregated_orderbook(symbol)
            if snap is None:
                return
            walls = compute_walls_buckets_v3(snap, top_n=8)
            if walls:
                get_manipulation_tracker().update(symbol, walls)
        except Exception as e:
            print(f"[LIQSIG] {symbol} heartbeat error: {e}")
    
    def _persist_scores(self):
        """Save the latest per-symbol scores to DB (compact JSON). Only
        the fields the UI badges need — full breakdowns stay in memory."""
        try:
            import json
            with self._lock:
                compact = {k: {'long': v['long_score'],
                                'short': v['short_score'],
                                'ts': v['ts']}
                           for k, v in self._last_scores.items()}
            self.db.set_setting('liqmap_signal_scores', json.dumps(compact))
        except Exception as e:
            print(f"[LIQSIG] persist scores error: {e}")
    
    def _load_persisted_scores(self) -> Dict:
        """Read the score snapshot from DB. Used when this process's
        in-memory map is empty (other-worker request or fresh restart).
        Entries older than 15 min are dropped — stale badges mislead."""
        try:
            import json
            raw = self.db.get_setting('liqmap_signal_scores', '{}')
            data = json.loads(raw) if raw else {}
            cutoff = time.time() - 900
            return {k: v for k, v in data.items()
                    if isinstance(v, dict) and v.get('ts', 0) >= cutoff}
        except Exception:
            return {}
    
    def get_status(self) -> Dict:
        with self._lock:
            scores = dict(self._last_scores)
        s = self._settings()
        if scores:
            scores_out = {k: {'long': v['long_score'], 'short': v['short_score'],
                               'ts': v['ts']} for k, v in scores.items()}
        else:
            # Cross-worker / post-restart fallback: read persisted snapshot
            scores_out = self._load_persisted_scores()
        return {
            'running': bool(self._thread and self._thread.is_alive()),
            'enabled': s['enabled'],
            'threshold': s['threshold'],
            'cooldown_min': s['cooldown_min'],
            'cycle_count': self._cycle_count,
            'scores': scores_out,
        }


_instance: Optional[LiqmapSignalScanner] = None


def get_liqmap_signal_scanner() -> Optional[LiqmapSignalScanner]:
    return _instance


def init_liqmap_signal_scanner(db=None, notifier=None) -> LiqmapSignalScanner:
    global _instance
    if _instance is None:
        _instance = LiqmapSignalScanner(db=db, notifier=notifier)
        _instance.start()
    return _instance
