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
FUEL_BAND_PCT = 0.15     # liq fuel counted within ±15% (distance-weighted)
FORECAST_STALE_MIN = 30  # forecast cache older than this = component absent
SQUEEZE_TFS = ('15m', '1h')
SQUEEZE_TF_WEIGHTS = {'15m': 0.45, '1h': 0.55}  # higher TF carries more


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
        # Staleness gate (2026-06-12): a forecast computed hours ago is
        # not a signal. If the cache is older than FORECAST_STALE_MIN the
        # component is treated as ABSENT and its weight is redistributed —
        # previously a stale/empty forecast silently zeroed 30 points and
        # systematically depressed every score (asymmetry vs liq).
        f1 = f4 = None
        forecast_fresh = False
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe is not None:
                cached = fe.get(symbol)
                if cached:
                    age_min = (time.time()
                               - (cached.get('computed_at') or 0)) / 60.0
                    if age_min <= FORECAST_STALE_MIN:
                        f1 = cached.get('forecast_1h')
                        f4 = cached.get('forecast_4h')
                        forecast_fresh = True
        except Exception:
            pass
        
        # --- Squeeze: MULTI-TIMEFRAME (2026-06-12) ---
        # 15m (entry lens) + 1h (structure lens). The 1h carries more
        # weight: higher-TF compression precedes the larger expansions.
        # Per-TF momentum sides are kept for directional modulation in
        # the composite (an aligned squeeze is worth more than one
        # coiling against you).
        sq_by_tf = {}
        try:
            from detection.market_data import get_market_data
            from detection.squeeze import calc_squeeze
            md = get_market_data()
            if md is not None:
                for tf in SQUEEZE_TFS:
                    kl = md.fetch_klines(symbol, interval=tf, limit=120)
                    if kl:
                        s = calc_squeeze(kl)
                        if s.get('ok'):
                            sq_by_tf[tf] = s
        except Exception:
            pass
        
        # --- Liquidation fuel: decayed LEVELS, profile-consistent ---
        # (2026-06-12) Reads the same decay profile the dashboard ladder
        # and the squeeze-probability endpoint use (DB setting), works on
        # per-level decayed USD, and weights each level by proximity:
        # 1/(1+dist%/2). Replaces the old cluster-zone ±5% raw sums —
        # those mixed day-old builds with live pools at equal weight.
        liq_above_w = liq_below_w = None
        liq_profile = 'tori'
        try:
            from detection.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm is not None:
                try:
                    liq_profile = self.db.get_setting(
                        'liqmap_decay_profile', 'tori') or 'tori'
                except Exception:
                    liq_profile = 'tori'
                state = lm.get_state(symbol=symbol, lookback_hours=24,
                                      profile=liq_profile)
                mark = state.get('mark_price')
                levels = state.get('levels') or []
                if mark and levels:
                    above = below = 0.0
                    for lev in levels:
                        dist_pct = abs(lev['price'] - mark) / mark * 100.0
                        if dist_pct > FUEL_BAND_PCT * 100:
                            continue
                        wgt = lev['usd'] / (1.0 + dist_pct / 2.0)
                        if lev['price'] > mark:
                            above += wgt
                        else:
                            below += wgt
                    if above + below > 0:
                        liq_above_w, liq_below_w = above, below
        except Exception:
            pass
        
        # =========================================================
        # Component sub-scores in [-1, +1] (positive favors LONG)
        # =========================================================
        # Forecast: blended side×confidence as on the dashboard
        forecast_sub = None
        if forecast_fresh:
            s1 = ((f1 or {}).get('side') or 0) * ((f1 or {}).get('confidence') or 0)
            s4 = ((f4 or {}).get('side') or 0) * ((f4 or {}).get('confidence') or 0)
            forecast_sub = max(-1.0, min(1.0, (s1 * 0.4 + s4 * 0.6) / 100.0))
        
        # Imbalance: ±30% book imbalance saturates the sub-score
        imb = walls.get('imbalance_pct') or 0
        imbalance_sub = max(-1.0, min(1.0, imb / 30.0))
        
        # Liq fuel: distance-weighted directional asymmetry, ×1.3 to
        # saturate (proximity weighting already shapes the magnitude)
        liq_sub = None
        if liq_above_w is not None:
            den = liq_above_w + liq_below_w
            liq_sub = (max(-1.0, min(1.0,
                       (liq_above_w - liq_below_w) / den * 1.3))
                       if den > 0 else 0.0)
        
        # Squeeze: multi-TF compression energy + per-side momentum
        # alignment. Energy E = Σ tf_weight × compression(tf) over TFs
        # currently ON. Alignment A(side) = Σ tf_weight over ON TFs whose
        # momentum points to `side`, normalized by total ON weight.
        # The composite later credits side s with E × (0.5 + 0.5·A(s)) —
        # a squeeze coiling WITH you is worth double one coiling against.
        squeeze_energy = 0.0
        sq_align = {1: 0.0, -1: 0.0}
        sq_on_any = False
        _on_w = 0.0
        for tf, tfw in SQUEEZE_TF_WEIGHTS.items():
            s = sq_by_tf.get(tf)
            if not (s and s.get('squeeze_on')):
                continue
            sq_on_any = True
            _on_w += tfw
            squeeze_energy += tfw * (s.get('probability') or 0) / 100.0
            mside = (1 if s.get('momentum', 0) > 0 else
                     -1 if s.get('momentum', 0) < 0 else 0)
            if mside:
                sq_align[mside] += tfw
        if _on_w > 0:
            sq_align = {k: v / _on_w for k, v in sq_align.items()}
        
        # Manipulation trust in [0, 1]: 0% spoofs → 1.0, ≥80% → 0.
        # pct is already USD-weighted + sample-shrunk by the tracker.
        trust = 1.0
        if manip and not manip.get('warming_up'):
            trust = max(0.0, 1.0 - (manip.get('pct', 0) / 80.0))
        
        # =========================================================
        # Directional composites — generalized weight redistribution
        # =========================================================
        # Base weights sum to 90 directional + 10 manip. ANY absent
        # component (stale forecast, no liq data, no squeeze) gives its
        # weight back proportionally to the present ones — no component's
        # absence silently depresses the score anymore.
        # REDISTRIBUTION CAP (×1.5 of base): without it a lone surviving
        # component inherited the full 90 and book imbalance alone could
        # fire an alert. With the cap, undistributed weight simply
        # vanishes — data poverty honestly lowers the score ceiling:
        #   4 components → 90 max · 3 → 90 · 2 → 60+10 (< поріг 75, тобто
        #   книга+squeeze без forecast і без liq алерт дати НЕ можуть).
        BASE_W = {'forecast': 30.0, 'imb': 20.0, 'liq': 20.0, 'squeeze': 20.0}
        present = {
            'forecast': forecast_sub is not None,
            'imb': True,
            'liq': liq_sub is not None,
            'squeeze': sq_on_any,
        }
        active_sum = sum(w for k, w in BASE_W.items() if present[k])
        W = {k: (min(w * 90.0 / active_sum, w * 1.5) if present[k] else 0.0)
             for k, w in BASE_W.items()}
        w_manip = 10.0
        
        def directional(sign: int) -> float:
            d_forecast = max(0.0, (forecast_sub or 0.0) * sign)
            d_imb = max(0.0, imbalance_sub * sign)
            d_liq = max(0.0, (liq_sub or 0.0) * sign)
            d_squeeze = squeeze_energy * (0.5 + 0.5 * sq_align.get(sign, 0.0))
            score = (W['forecast'] * d_forecast
                     + W['imb'] * d_imb
                     + W['liq'] * d_liq
                     + W['squeeze'] * d_squeeze)
            # Manipulation scales the whole conviction; its own weight is
            # granted only at full trust + at least one directional vote
            score = score * (0.5 + 0.5 * trust) + w_manip * trust * (
                1 if (d_forecast > 0 or d_imb > 0 or d_liq > 0) else 0)
            return max(0.0, min(100.0, score))
        
        sq15 = sq_by_tf.get('15m') or {}
        sq1h = sq_by_tf.get('1h') or {}
        breakdown = {
            'symbol': symbol,
            'ts': time.time(),
            'mid_price': walls.get('mid_price'),
            'long_score': round(directional(+1), 1),
            'short_score': round(directional(-1), 1),
            'forecast_sub': (round(forecast_sub, 3)
                             if forecast_sub is not None else None),
            'forecast_fresh': forecast_fresh,
            'imbalance_pct': round(imb, 2),
            'liq_above_usd': liq_above_w,     # distance-weighted, decayed
            'liq_below_usd': liq_below_w,
            'liq_profile': liq_profile,
            'liq_sub': (round(liq_sub, 3) if liq_sub is not None else None),
            'squeeze_on': sq_on_any,
            'squeeze_energy': round(squeeze_energy, 3),
            'squeeze_tfs': {tf: {'on': bool(s.get('squeeze_on')),
                                  'prob': s.get('probability', 0),
                                  'band': s.get('band', 'off'),
                                  'bars': s.get('bars_in_squeeze', 0),
                                  'mom': (1 if s.get('momentum', 0) > 0
                                          else -1 if s.get('momentum', 0) < 0
                                          else 0)}
                            for tf, s in sq_by_tf.items()},
            'trust': round(trust, 2),
            'weights': {k: round(v, 1) for k, v in W.items()},
            'manip_pct': (manip or {}).get('pct') if manip and not manip.get('warming_up') else None,
            'manip_low_sample': bool((manip or {}).get('low_sample')),
            'sources': walls.get('sources') or [],
            'f1': f1, 'f4': f4,
            # legacy fields some consumers may still read
            'squeeze_prob': sq15.get('probability', 0) or sq1h.get('probability', 0),
            'squeeze_bars': sq15.get('bars_in_squeeze', 0),
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
        
        def fu(v):
            if v is None:
                return '—'
            return f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K"
        
        W = br.get('weights') or {}
        lines = [
            f"💧 <b>LIQUIDITY MAP SIGNAL</b>",
            f"{emoji} <b>{side} {sym}</b> @ {price_s} · score <b>{score:.0f}</b>/100 (поріг {threshold})",
            "",
        ]
        # Component table — each line: value + its weight in this score
        # (weights vary per symbol because absent components redistribute)
        f1, f4 = br.get('f1') or {}, br.get('f4') or {}
        def fdesc(f):
            s = f.get('side') or 0
            if not s:
                return '—'
            return f"{'LONG' if s > 0 else 'SHORT'} {f.get('confidence', 0)}%"
        if br.get('forecast_fresh'):
            lines.append(f"🔮 Forecast (w{W.get('forecast', 0):.0f}): "
                         f"1H {fdesc(f1)} · 4H {fdesc(f4)}")
        else:
            lines.append("🔮 Forecast: застарілий — вага перерозподілена")
        lines.append(f"⚖️ Imbalance (w{W.get('imb', 0):.0f}): "
                     f"{br['imbalance_pct']:+.1f}% "
                     f"({'+'.join(br.get('sources') or []) or 'book'})")
        if br.get('liq_above_usd') is not None:
            lines.append(
                f"🧲 Liq fuel (w{W.get('liq', 0):.0f}, "
                f"профіль {br.get('liq_profile', 'tori')}): "
                f"↑{fu(br['liq_above_usd'])} / ↓{fu(br['liq_below_usd'])} "
                f"(±15%, дистанційно зважено)")
        else:
            lines.append("🧲 Liq fuel: нема даних — вага перерозподілена")
        sq_tfs = br.get('squeeze_tfs') or {}
        on_tfs = [(tf, s) for tf, s in sq_tfs.items() if s.get('on')]
        if on_tfs:
            parts = [f"{tf} {s.get('band', '?')} {s.get('prob', 0):.0f}%"
                     f" mom{'↑' if s.get('mom', 0) > 0 else '↓' if s.get('mom', 0) < 0 else '·'}"
                     for tf, s in sorted(on_tfs)]
            lines.append(f"⚡ Squeeze (w{W.get('squeeze', 0):.0f}): "
                         + ' · '.join(parts))
        else:
            lines.append("⚡ Squeeze: OFF на 15m і 1h")
        if br.get('manip_pct') is not None and not br.get('manip_low_sample'):
            lines.append(f"🎭 Manipulation: {br['manip_pct']:.0f}% "
                         f"(USD-зважено) · trust {br.get('trust', 1):.2f}")
        else:
            lines.append(f"🎭 Manipulation: статистика накопичується "
                         f"· trust {br.get('trust', 1):.2f}")
        lines.append("")
        lines.append("⏱ Підтверди тайминг входу через SMC-структуру (CHoCH/BOS)")
        
        try:
            from alerts.telegram_notifier import get_notifier
            tg = get_notifier()
            if tg and tg.enabled:
                tg.send_message('\n'.join(lines))
                print(f"[LIQSIG] Alert sent: {side} {sym} score={score:.0f}")
        except Exception as e:
            print(f"[LIQSIG] Telegram send error: {e}")

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
                compact = {}
                for k, v in self._last_scores.items():
                    sq_tfs = v.get('squeeze_tfs') or {}
                    compact[k] = {
                        'long': v['long_score'],
                        'short': v['short_score'],
                        'ts': v['ts'],
                        # brief breakdown — feeds watchlist button tooltips
                        'brk': {
                            'fc': v.get('forecast_sub'),
                            'im': v.get('imbalance_pct'),
                            'lq': v.get('liq_sub'),
                            'pf': v.get('liq_profile'),
                            'sq': v.get('squeeze_energy'),
                            'sqtf': {tf: s.get('prob', 0)
                                     for tf, s in sq_tfs.items()
                                     if s.get('on')},
                            'tr': v.get('trust'),
                        },
                    }
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
            scores_out = {}
            for k, v in scores.items():
                sq_tfs = v.get('squeeze_tfs') or {}
                scores_out[k] = {
                    'long': v['long_score'], 'short': v['short_score'],
                    'ts': v['ts'],
                    'brk': {
                        'fc': v.get('forecast_sub'),
                        'im': v.get('imbalance_pct'),
                        'lq': v.get('liq_sub'),
                        'pf': v.get('liq_profile'),
                        'sq': v.get('squeeze_energy'),
                        'sqtf': {tf: s.get('prob', 0)
                                 for tf, s in sq_tfs.items() if s.get('on')},
                        'tr': v.get('trust'),
                    },
                }
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
