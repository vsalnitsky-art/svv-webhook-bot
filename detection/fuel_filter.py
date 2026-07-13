"""
fuel_filter — server-side automated trading filter driven by liquidation
"fuel" direction.

Idea (operator spec):
  • Scan every WATCHLIST coin.
  • When a coin shows "Паливо зверху (тягне в LONG)" (fuel_dir > +0.1) or
    the opposite "Паливо знизу (тягне в SHORT)" (fuel_dir < -0.1) a per-coin
    TIMER starts.
  • If that same status holds continuously for ≥ duration_minutes
    (default 5) we OPEN a position in the fuel direction.
  • The position stays open until EITHER:
       – the fuel status changes / disappears (timer resets) → CLOSE, or
       – (optional, toggled) the move "Потенціал"/exhaustion reaches
         potential_threshold_pct (default 95%) → CLOSE (take the move
         before it reverses).

Safety / exchange-friendliness:
  • Fuel direction is read from the *cached* liquidation map state
    (refreshed by its own daemon every 60 s) — this module makes ZERO extra
    exchange calls to compute fuel.
  • IMPORTANT: the liq-map daemon only tracks BACKGROUND_SYMBOLS (BTC/ETH)
    plus symbols requested on-demand in the last 30 min. So each tick we
    call lm.request_symbol() for every WATCHLIST coin — this is what makes
    the daemon actually scan ALL watchlist coins (not just the ones a user
    happens to be viewing in the UI). First-seen coins need ~1-2 liq-map
    ticks (≤2 min) before fuel data becomes meaningful.
  • Exhaustion ("Потенціал") needs klines, so it is computed ONLY for coins
    that already have an OPEN fuel position (a handful at most), never for
    the whole watchlist. Results are cached per-symbol with a short TTL.
  • The scan cycle is 30 s — twice per liq-map refresh, which is plenty.

Persistence / recovery:
  • All live state (per-coin timers, open positions, recent closes) and all
    settings are persisted to the DB as JSON settings, mirroring how the
    Trade Manager persists its book. On boot the daemon restores everything
    so timers and positions survive restarts/redeploys.

Execution modes:
  • Positions are opened via Trade Manager (real Bybit) or Test Mode (paper)
    based on their toggle states — this filter doesn't maintain its own mode.
  • If TM is enabled → real position via manual_open()
  • If Test Mode is enabled → paper position via _open_shadow()
  • If both are on, real TM takes precedence.
"""

import time
import threading
from typing import Optional, Callable, Dict, List

CYCLE_SECS = 30                 # scan cadence (twice per liq-map refresh)
EXHAUSTION_TTL = 120            # cache exhaustion per symbol for 2 min
BIAS_TTL = 10                   # cache compute_bias result per symbol (sec)
FUEL_LONG_THR = 0.1            # fuel_dir > +0.1 → LONG bias
FUEL_SHORT_THR = -0.1          # fuel_dir < -0.1 → SHORT bias
CLOSED_LIMIT = 100             # keep last N closes for the UI
# Grace period before closing on FUEL FADE (status → neutral/None). Without
# this, a single transient liq-map data gap or a brief dip into the ±0.1
# neutral zone would slam the position shut the very next tick. We only honour
# a *sustained* loss of fuel. A clear FLIP to the opposite side still closes
# immediately (that's a real reversal, not a data gap).
FUEL_FADE_GRACE_SEC = 180      # 3 min of continuous neutral before close
# Minimum time a freshly-opened position is held before ANY auto-exit
# (flip / fade / wait / exhaustion) may fire. A manual force-open takes the
# timer's side regardless of where fuel currently points, so without this
# grace the very next tick could see fuel on the opposite side and close the
# trade instantly ("opened → immediately in Recent Closed Trades").
MIN_HOLD_AFTER_OPEN_SEC = 90

_DB_SETTINGS = 'fuel_filter_settings'
_DB_STATE = 'fuel_filter_state'
_DB_SCAN_LIST = 'fuel_filter_scan_list'   # which symbols FF is allowed to scan
_DB_FUNDING_ARCHIVE = 'fuel_filter_funding_archive'  # per-coin funding history
FUNDING_ARCH_MAX = 800          # max samples kept per coin
FUNDING_ARCH_COINS_MAX = 400    # max distinct coins archived
FUNDING_ARCH_SAMPLE_SEC = 60    # periodic sample cadence while a coin is live

# ─── queue (❤️ Черга на вхід) removal operations ───
# Every place that REMOVES a coin from _pending is numbered and guarded. These
# were temporarily frozen (all False) to diagnose coins vanishing from the
# queue. The real culprit was the queue being RESTORED from the DB on boot
# (stale coins like AAVEUSDT reappearing without a fresh signal) — now fixed by
# making the queue ephemeral in _load_state. All removal ops are verified
# correct, so they are re-enabled (True). Flip a value to False to re-freeze a
# single op for future debugging.
#   1 = position closed (_close)
#   2 = engine opened the trade (_engine_tick success)
#   3 = engine: TM already holds the coin (_engine_tick)
#   4 = ММ-flip purge of opposite-direction entries (_update_btc_verdict)
#   5 = manual delete ✕ (delete_timer)
#   6 = manual "Очистити всі" (clear_all_timers)
#   7 = remove_pending()
#   8 = clear_pending()
#   9 = manual force-open (force_open_timer success)
#  10 = TTL expiry — coin waited longer than queue_ttl_hours without opening
_QUEUE_OPS_ALLOWED = {1: True, 2: True, 3: True, 4: True, 5: True,
                      6: True, 7: True, 8: True, 9: True, 10: True}


def _q_allowed(op: int) -> bool:
    """True if queue-removal op #op is currently allowed (debug gate)."""
    if _QUEUE_OPS_ALLOWED.get(op, True):
        return True
    print(f"[FF-QUEUE] op#{op} BLOCKED — removal suppressed (coin kept in queue)")
    return False

DEFAULT_SETTINGS = {
    'enabled': False,
    'duration_minutes': 5,        # min duration to show in table (threshold filter)
    'potential_threshold_pct': 95,  # exhaustion ≥ this → close
    'use_potential_exit': True,   # toggle the exhaustion exit on/off
    'max_exhaustion_pct': 75,     # engine/open GATE: skip a coin if its move is
                                  # more exhausted than this % (0..100). Active.
    # ❤️ queue TTL: a coin that has waited longer than this many HOURS without
    # opening is auto-dropped from the queue (OP-10). Keeps the now two-sided
    # queue from growing forever. 0 = off (only manual/flip removal).
    'queue_ttl_hours': 24,
    'skip_wait_coins': False,     # (legacy) not used for auto-open anymore
    'manage_open_positions': True,  # if True, FF closes positions it opened
    # Auto-close an open (real OR test) position when its ММ (fuel) STRENGTH
    # falls below this % (|fuel dir|×100). 0 = off. Works only while FF manages
    # the position (manage_open_positions=True).
    'manage_close_min_mm': 0,
    # Auto-close trades vs the ₿ BTCUSDT banner. Mode:
    #   'off'   — вимкнено;
    #   'pause' — закривати вже коли банер втратив підтвердження (ПАУЗА/WAIT
    #             по стороні угоди) АБО розвернувся — раніше й агресивніше;
    #   'flip'  — лише коли банер ПОВНІСТЮ розвернувся у протилежний бік.
    # close_on_btc_flip лишено як legacy-фолбек (True == 'flip').
    'close_on_btc_mode': 'off',
    'close_on_btc_flip': False,
    'direction_smoothing_min': 0,   # EMA window (min) for ММ direction; 0 = OFF (raw)
    'anomaly_hours': 10,            # fuel held longer than this → "anomaly" list
    'start_signal_minutes': 5,      # BTC ММ held ≥ this → START signal (else STOP)
    # ── BTC-START auto-engine (banner toggle) ──
    'start_engine_enabled': False,        # master: auto-open basket on BTC START
    'start_engine_independent': False,    # auto-open independent of BTC (own dir)
    'start_engine_use_anomalies': True,   # source: 🜂 anomalies table
    'start_engine_use_timers': True,      # source: ⏱️ active timers table
    'start_engine_scan_secs': 15,         # engine scan cadence (sec)
    'start_engine_include_funding': False,# include 💰 funding-marked coins
    'engine_candle_confirm': True,        # only open when candles confirm dir (2/2)
    'engine_candle_tf': '5m',             # timeframe for the candle confirmation
    'engine_require_strong_hold': False,  # only open when SCORE=STRONG HOLD & dir matches
    # 🧭 Smart direction: ignore the queued signal side; derive each coin's
    # direction from its LIVE fuel (ММ) instead, then apply ALL the usual FF
    # gates for THAT direction. Rescues coins that entered on the "wrong" signal
    # while every indicator favours the opposite side.
    'engine_smart_direction': False,
    # Minimum ММ (fuel) STRENGTH % to OPEN a trade — SEPARATE per direction.
    # 0 = off, 30 = помірний (≥30%), 60 = сильне (≥60%). The engine skips a
    # candidate whose fuel strength (|fuel dir|×100) is below the threshold
    # for its direction. `engine_min_mm_strength` kept as legacy fallback.
    'engine_min_mm_strength': 0,
    'engine_min_mm_strength_long': 0,
    'engine_min_mm_strength_short': 0,
    # Min Decision-Center verdict to OPEN: 'any' | 'marginal' | 'good'.
    # 'marginal' = МЕЖОВИЙ або кращий; 'good' = лише ДОБРИЙ. Оцінюється
    # лениво лише для кандидата, що вже пройшов усі інші гейти.
    'engine_min_decision': 'any',
    # ── ⚡ CTR (Schaff Trend Cycle) OPEN gate ──
    # Uses CTR as an ENTRY-TIMING filter for the momentum-continuation FF entry.
    #   'off'          — no CTR gate.
    #   'anti_extreme' — block LONG when STC overbought (≥ threshold), block
    #                    SHORT when STC oversold (≤ 100−threshold). Avoids
    #                    buying tops / selling bottoms. (recommended base)
    #   'fresh_cross'  — require a fresh CTR crossover in the trade direction
    #                    (last_dir == dir AND age ≤ max_age_bars).
    #   'both'         — anti_extreme AND fresh_cross.
    'ctr_gate_mode': 'off',
    # «Не входити проти нахилу»: min reversal-lean % (|STC−50|/50·100, same as the
    # CTR column) that blocks an opposite-side trade. 50 ≈ old STC-threshold 75.
    'ctr_gate_lean_pct': 50,
    'ctr_gate_stc_threshold': 75,         # (legacy) raw STC level — superseded by lean%
    'ctr_gate_max_age_bars': 3,           # fresh-crossover max age (bars)
    # ── Two independent entry queues (both fed by CHoCH/CHoCH+BOS signals) ──
    #   Queue 1 «❤️ Черга на вхід» — classic interception (buttons/₿ START gated).
    #   Queue 2 «⚡ CTR-зони»       — holds the signal until SCORE + CTR both align
    #     with the signal direction, then opens — INDEPENDENT of bars & buttons.
    # Both OFF → signals are NOT intercepted and open directly per their own flow.
    'queue1_enabled': True,
    'queue2_enabled': False,
    # ── Queue 2 eject rules (its own settings accordion in the UI) ──
    #   queue2_eject_ctr      — drop a QUEUED coin when the CTR lean turns to the
    #     OPPOSITE side by at least queue2_eject_ctr_pct % (|STC−50|/50·100).
    #     Default OFF (the coin just waits on an opposite/neutral CTR otherwise).
    #   queue2_eject_choch    — drop a QUEUED coin when an OPPOSITE-direction
    #     CHoCH/CHoCH+BOS for the SAME coin arrives (the old signal is stale).
    #     Default ON. Fires even when the new opposite signal itself isn't queued
    #     (e.g. its CTR gate dropped it or it went straight to a trade).
    'queue2_eject_ctr': False,
    'queue2_eject_ctr_pct': 20,
    'queue2_eject_choch': True,
    #   queue2_use_buttons    — when ON, Queue 2 opens only for a direction that
    #     the main LONG/SHORT buttons allow (like Queue 1). Default OFF → Queue 2
    #     opens independently of the buttons.
    'queue2_use_buttons': False,
    #   queue2_use_btc        — when ON, Queue 2 opens ONLY in the committed ₿
    #     BTCUSDT session direction (the banner): a LONG signal opens only while
    #     the ₿ session is LONG (active, not paused), SHORT only while ₿ is SHORT.
    #     ₿ WAIT/ПАУЗА or no direction → hold (don't open). Composes (AND) with
    #     queue2_use_buttons.
    #   CALIBRATED 2026-07-13: default OFF→ON. LONG trades returned -1.32%/20%win
    #     vs SHORT +2.75%/50% in a down-trend — an ADAPTIVE ₿-direction gate
    #     suppresses the losing counter-trend side (regime-adaptive, NOT a
    #     hard-coded side). Turn OFF to open both sides regardless of ₿.
    'queue2_use_btc': True,
    #   queue2_open_min_dec_score — minimum mature Decision-Center score (side-
    #     specific, −100..100) to OPEN from Queue 2. Runs LAST (after the cheap
    #     gates) so compute_decision is rare. CALIBRATED 2026-07-13: profitable
    #     band was dec ≥ 40 (+2.79%), below 40 flat-to-negative; default 20 is a
    #     BALANCED floor (drops the losing [0,20) bucket, keeps enough trades
    #     ≈ +1.44% exp). Raise toward 40 for the data-optimal. 0 = disabled.
    'queue2_open_min_dec_score': 20,
    #   queue2_hold_unknown_ctr — when CTR data is NOT yet available at signal
    #     time (lean = «—»), HOLD the signal in Queue 2 instead of dropping it,
    #     and wait for CTR to appear (the engine opens/ejects once it does).
    #     Default ON — «невідомо ≠ проти», щоб не втрачати сигнали на холодному
    #     кеші CTR. OFF = old behaviour (unknown CTR → drop).
    'queue2_hold_unknown_ctr': True,
    #   queue2_queue_all — when ON, Queue 2 QUEUES EVERY incoming CHoCH/CHoCH+BOS
    #     regardless of the CTR lean at signal time (even when CTR is OPPOSITE),
    #     and holds it until SCORE=STRONG HOLD & CTR align (the engine opens then)
    #     — instead of dropping it outright. Catches signals whose CTR flips in
    #     their favour a few minutes later. Default OFF (drop on opposite CTR).
    #     Pair with TTL / «викид за протилежним CTR» to clear ones that never come
    #     good.
    'queue2_queue_all': False,
    #   queue2_reverse_via_queue — an OPEN OPPOSITE position may be reversed by an
    #     opposite signal ONLY after that signal FULLY passes Queue 2 (SCORE=STRONG
    #     HOLD + CTR aligned). Such a signal is QUEUED (not reversed on arrival);
    #     the Q2 engine closes the opposite trade and opens the new one only when
    #     it's ready to open. Also suppresses the raw-CHoCH reverse-close so the
    #     reversal is gated by Q2, not by the bare signal. Default ON.
    'queue2_reverse_via_queue': True,
    # ── Queue 2 PROFESSIONAL entry algorithm (hold-and-wait) ──
    #   queue2_hold_and_wait — a fresh CHoCH fires AGAINST the CTR lean by nature,
    #     so dropping on opposite CTR at signal time throws away ~76% of setups
    #     (open-rate ~2%). Instead: QUEUE every QUALITY signal (ENTRY score ≥
    #     min), regardless of CTR direction, and open it once CTR pulls into
    #     alignment (a reversal-pullback entry). Default ON — the professional
    #     path. OFF = legacy drop-on-opposite-CTR.
    'queue2_hold_and_wait': True,
    #   queue2_ctr_neutral_pct — CTR dead-zone. A lean whose strength
    #     (|STC−50|/50·100) is BELOW this counts as NEUTRAL, not «against». Stops
    #     a trivial 3% lean from rejecting a strong setup. Matches the ±0.1 band
    #     used for the ₿ verdict.
    'queue2_ctr_neutral_pct': 10,
    #   queue2_min_entry_score — minimum ENTRY score (0..100, setup quality — NOT
    #     the hold score) required to ENTER Queue 2. Keeps ВИСНАЖЕНО/СЛАБКЕ trash
    #     out of the queue.
    'queue2_min_entry_score': 45,
    #   queue2_open_min_entry_score — minimum live ENTRY score required to OPEN
    #     from Queue 2 (with CTR aligned/neutral). Higher than the queue-in bar so
    #     only setups that stayed strong actually fire.
    #   CALIBRATED 2026-07-13 (90 closed paper trades): 60→65. ENTRY buckets showed
    #     [55,65)=-0.79%/20%win, [65,75)=+1.25%/31%, and filtering ff_entry≥65
    #     lifted expectancy +0.216%→+0.700% (sum +19→+45). NOTE: ≥70 OVERSHOOTS
    #     (turns NEGATIVE, -0.205%) — 65 is the sweet spot, do NOT raise further
    #     without fresh data.
    'queue2_open_min_entry_score': 65,
    #   queue2_ttl_hours — a queued signal expires after this many hours (logged
    #     «протерміновано») instead of lingering until an opposite CHoCH ejects it.
    #   CALIBRATED 2026-07-13: 6→2. Trades that waited >2h in queue returned
    #     -1.28% (vs +1.29% for <5min); capping wait≤2h lifted expectancy to
    #     +0.927%. 0 = no TTL.
    'queue2_ttl_hours': 2,
    #   ctr_mtf — multi-timeframe CTR confluence (RECORD-ONLY for now, does NOT
    #     gate). A fresh CHoCH is contrarian to the single 15m CTR by nature;
    #     reading STC across TFs separates TREND context (1h/4h) from ENTRY timing
    #     (5m/15m) so we can later prove whether multi-TF alignment predicts PnL
    #     better than the single 15m CTR — then optionally gate on it.
    'ctr_mtf_enabled': True,
    'ctr_mtf_tfs': ['5m', '15m', '45m', '1h', '4h'],
    #   q2_auto_ob_sl — auto-manage each open trade's Manual SL from the nearest
    #     «Require OB Match» Order Block: LONG → OB low − buffer%, SHORT → OB high
    #     + buffer%. Ratchets tighter as a BETTER (closer) OB forms; never loosens.
    #     Default OFF. Applies to real + paper open positions.
    'q2_auto_ob_sl': False,
    'q2_auto_ob_sl_buffer_pct': 0.2,
    #   q2_auto_ob_sl_tf — OB-таймфрейм САМЕ для авто-Manual-SL (незалежний від
    #     ob_filter_timeframe сканера). За замовч. '15m' — головний скан-TF, для
    #     якого OB рахується завжди. Інші TF працюють лише якщо сканер їх теж
    #     обробляє (ob_filter / pd_zone / exit TF), інакше OB-рядка не буде.
    'q2_auto_ob_sl_tf': '15m',
    'start_signal_tg_alerts': False,      # Telegram alert on BTC START/STOP change
    'funding_duration_minutes': 0,        # separate show-threshold for 💰 funding coins
    'funding_tg_alerts': False,           # Telegram alert when a funding coin enters the table
    # A funding coin appears in the 💰 ММ table only if its ММ (fuel) STRENGTH
    # (|fuel dir|×100) ≥ this. 0 = off, 30 = помірний (≥30%), 60 = сильне (≥60%).
    'funding_min_mm_strength': 0,
    # Extra filter for the ENTRY ("appear") Telegram message ONLY (does NOT
    # affect the table, and does NOT affect exit messages):
    #   funding_tg_entry_dir     — 'any' | 'LONG' | 'SHORT' (тільки цей напрямок)
    #   funding_tg_entry_min_mm  — мін. сила ММ, щоб слати «появу» (0 = off)
    'funding_tg_entry_dir': 'any',
    'funding_tg_entry_min_mm': 0,
    # Anti-spam for funding-coin appear alerts:
    #  cooldown — не слати повторну «появу» по монеті стільки хвилин;
    #  hysteresis — монета «зникає» лише коли сила ММ впаде на стільки % НИЖЧЕ
    #  порога входу (тремтіння рівно на межі не робить churn).
    'funding_notify_cooldown_min': 30,
    'funding_mm_hysteresis': 10,
    # Optional per-coin SESSION mode (like the ₿ banner). When ON, a funding
    # coin's ММ direction commits into a session: WAIT = ПАУЗА (keeps the coin,
    # no re-announce), opposite flip = NEW session (single re-announce). Kills
    # spam far harder than cooldown/hysteresis. Off = classic behaviour.
    'funding_session_mode': False,
    # ── Telegram про 💰 funding-монети: поява/зникнення в таблиці ММ ──
    # Replaces the old funding alert. Fires when a funding coin APPEARS
    # (ff_tg_on_entry) / DISAPPEARS (ff_tg_on_exit) from the 💰 ММ table.
    # Templates support {symbol} {dir} {side} {price} {funding} {funding_in}
    # {fuel} {exhaustion} {reason} {btc}. Missing placeholders render as "—".
    # {funding} = ставка фандінгу у %, {funding_in} = час до наступного фандінгу.
    'ff_tg_on_entry': False,
    'ff_tg_on_exit': False,
    'ff_tg_entry_template': '🚀 FF вхід {side} {symbol} ·💲 {price} ·\nММ {fuel}% фандінг {funding}% · 🔄 {funding_in}',
    'ff_tg_exit_template': '☄️ {symbol} зникла з 📡 · {reason}\n💲 {price} · 📡 {fuel}%',
}


class FuelFilterDaemon:
    def __init__(self, db, get_trade_manager: Callable,
                 get_watchlist: Callable):
        self._db = db
        self._get_tm = get_trade_manager
        self._get_watchlist = get_watchlist
        self._thread: Optional[threading.Thread] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._alert_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # live state (restored from DB on boot)
        self._timers: Dict[str, Dict] = {}      # symbol -> {dir, since}
        # Tracking dict: which symbols fuel filter opened (not full position data)
        self._fuel_managed: Dict[str, Dict] = {}  # symbol -> {opened_at, side, fuel_dir}
        # exhaustion cache: symbol -> {ts, exhaustion, side}
        self._exh_cache: Dict[str, Dict] = {}
        # compute_bias cache: symbol -> {ts, data}. Shared by exhaustion,
        # verdict and panel-status so we call the (heavy) shared bias
        # computation at most once per symbol per BIAS_TTL window.
        self._bias_cache: Dict[str, Dict] = {}
        # Decision-Center verdict cache (symbol → {ts, v}); used by the engine's
        # final open gate (engine_min_decision).
        self._decision_cache: Dict[str, Dict] = {}
        # Aggregate scan stats for the panel header: how many scan-targets were
        # scanned, watchlist size, and average move exhaustion (LONG/SHORT).
        # Recomputed each tick.
        self._scan_stats: Dict = {}
        # Pre-computed SCORE verdicts {sym: dict}, refreshed in the background
        # _tick so the UI endpoints (FF state + TM state) read them instantly
        # instead of doing per-row liq-map / kline-fetch work in the request
        # path (that made the page slow to open).
        self._score_cache: Dict[str, Dict] = {}
        # Fuel STRENGTH (0..100) per symbol — current cycle and previous cycle,
        # so the UI can show the value + a rising/falling trend. Refreshed once
        # per cycle in _refresh_score_cache (cheap; read by both tables).
        self._fuel_str: Dict[str, int] = {}
        self._fuel_str_prev: Dict[str, int] = {}
        # Symbols pulled in from the 💰 Funding Rate Scanner (when it's enabled).
        # They get fuel timers + a row in the ❤️ table, flagged distinctly, but
        # are MONITOR-ONLY (no auto-open / management). Refreshed each tick.
        self._funding_syms: set = set()
        # {symbol: ts of last funding "appear" TG} — anti-spam cooldown so a coin
        # that flickers around the ММ-strength threshold isn't re-announced.
        self._funding_notify_at: Dict[str, float] = {}
        # Persistent per-coin funding archive: {sym: [ {t,ev,mm,mm_dir,funding,
        # price,vol24h,btc_dir,btc_paused} ]}. Re-appearing coins APPEND to the
        # existing series. _funding_prev_state tracks last snapshot for event
        # detection; _funding_arch_last throttles periodic samples.
        self._funding_archive: Dict[str, List[Dict]] = {}
        self._funding_prev_state: Dict[str, Dict] = {}
        self._funding_arch_last: Dict[str, float] = {}
        self._funding_arch_dirty = False
        self._funding_arch_saved_at = 0.0
        # Per-coin MUTE: {sym: until_ts}. A muted coin is ignored entirely by
        # the 💰 Funding — ММ scan (no table row, no TG, no archive) until then.
        self._funding_muted: Dict[str, float] = {}
        # {symbol: current funding %} for funding-sourced coins (UI display).
        self._funding_rates: Dict[str, float] = {}
        # {symbol: nextFundingTime ms} — countdown to next funding settlement.
        self._funding_next: Dict[str, int] = {}
        # {symbol: 24h volume USD} for funding-sourced coins (UI display).
        self._funding_vols: Dict[str, float] = {}
        # Short-TTL klines cache for candle confirmation at a custom TF:
        # {(symbol, tf): (ts, klines)}.
        self._candle_cache: Dict = {}
        # {symbol: count} — how many times the engine tried to open this coin but
        # the candle confirmation failed (it stayed in the table waiting).
        self._engine_attempts: Dict[str, int] = {}
        # ── Diagnostics: WHY the engine didn't open a coin this tick ──
        # {symbol: human UA reason} — surfaced per-row in the ❤️ queue so the
        # operator sees exactly which gate blocked each coin (e.g. «паливо не
        # SHORT», «виснаж 88%>75%», «рішення СЛАБКИЙ»). Cleared/refreshed each
        # engine tick.  _engine_gate = a GLOBAL reason when the engine as a whole
        # can't open anything right now (двигун вимкнено / BTC пауза / чекає
        # START / кнопки L/S вимкнені).  Both are purely informational.
        self._engine_skip: Dict[str, str] = {}
        self._engine_gate: str = ''
        # Active engine mode: 'off' (FF disabled) | 'btc' (₿ START session) |
        # 'buttons' (main-banner LONG/SHORT). Surfaced so the UI announces HOW
        # the engine is working right now.
        self._engine_mode: str = 'off'
        # Direction smoothing state (anti-twitch): EMA of the raw liq-fuel
        # imbalance per symbol, plus the hysteresis direction latch. Advanced
        # ONCE per scan cycle in _tick; everyone else reads it. Persisted so
        # smoothing survives a restart instead of snapping back to raw.
        self._fuel_ema: Dict[str, float] = {}
        self._fuel_hyst: Dict[str, Optional[str]] = {}  # sym → 'LONG'/'SHORT'/None
        # Last BTC START/STOP state we sent a Telegram alert for (None until set).
        self._btc_start_last_alert: Optional[str] = None
        # Last live BTC direction — reported in the STOP message.
        self._btc_last_dir: Optional[str] = None
        # Funding coins already TG-alerted as "entered the table" (alert once).
        self._funding_alerted: set = set()
        # Latest BTC fuel snapshot — pinned permanently at the top of the table.
        self._btc_state: Dict = {}
        # ── BTC ММ "session" ──────────────────────────────────────────────
        # A session = a committed BTC ММ direction (LONG/SHORT). WAIT (ML
        # балансований) is a PAUSE — it keeps the session, its start time and
        # the queue. Only an OPPOSITE flip (LONG↔SHORT) ends the session, resets
        # the timer and CLEARS the queue. `_btc_verdict_dir`/`_btc_verdict_since`
        # hold the SESSION (not the live blip); `_btc_paused` is True while the
        # live ML is WAIT inside a live session (banner shows "напрямок · пауза").
        self._btc_verdict_dir: Optional[str] = None   # session dir 'LONG'/'SHORT'/None
        self._btc_verdict_since: float = 0.0           # session start (persists through pause)
        self._btc_paused: bool = False                 # live ML is WAIT within the session
        self._btc_fuel_strength: int = 0               # BTC fuel strength 0..100 (banner bar)
        # 💰 Funding fuel table (repurposed from the old anomalies storage):
        # coins from the 💰 Funding Rate Scanner that currently show fuel. Same
        # dict shape so the existing table/endpoints keep working. {symbol:
        # {dir, started_at, start_price, holding, last_price, last_held_sec,
        # ended_at, end_price}}
        self._anomalies: Dict[str, Dict] = {}
        # ❤️ FF "база" — intercepted coins WAITING to open. Populated by
        # intercept() (every coin the main LONG/SHORT flow would open while FF
        # is on), direction = the signal side (self-overwrites on re-entry).
        # On ₿ START the engine fuel-checks these and opens the matching ones.
        # {symbol: {dir, added_at}}. Persisted.
        self._pending: Dict[str, Dict] = {}
        # ── Queue 2 «⚡ CTR-зони» (reversion): SAME machinery/gates as _pending,
        # but coins ENTER by a CTR scan (STC ≤ low → LONG, STC ≥ high → SHORT)
        # instead of CHoCH/BOS interception. Independent on/off toggle. {sym:
        # {dir, added_at}}. Persisted.
        self._pending2: Dict[str, Dict] = {}
        self._last_tick_ts = 0
        self._load_state()

    def _get_funding_symbols(self) -> List[str]:
        """Tracked coins from the 💰 Funding Rate Scanner — but only while that
        module is enabled and running. Returns [] otherwise. These are merged
        into fuel scanning for monitoring only."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if not fm or not fm.is_enabled() or not getattr(fm, '_running', False):
                return []
            if hasattr(fm, 'get_symbols'):
                return [s.upper() for s in fm.get_symbols()]
            return []
        except Exception:
            return []

    def _get_funding_rates(self) -> Dict[str, float]:
        """{SYMBOL: current funding %} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_rates'):
                return {str(k).upper(): v for k, v in fm.get_rates().items()}
        except Exception:
            pass
        return {}

    def _get_funding_next(self) -> Dict[str, int]:
        """{SYMBOL: nextFundingTime ms} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_next_funding'):
                return {str(k).upper(): int(v) for k, v in fm.get_next_funding().items()}
        except Exception:
            pass
        return {}

    def _get_funding_volumes(self) -> Dict[str, float]:
        """{SYMBOL: 24h volume (USD)} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_volumes'):
                return {str(k).upper(): float(v) for k, v in fm.get_volumes().items()}
        except Exception:
            pass
        return {}

    def _get_funding_threshold(self) -> float:
        """The 💰 Funding Rate Scanner 'Entry ≤' threshold (negative %), used as
        the START of the funding progress bar (entry → -4)."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm is not None:
                return float(getattr(fm, '_entry_threshold', -1.0))
        except Exception:
            pass
        return -1.0

    # ------------------------------------------------------------------
    # settings (DB-persisted JSON)
    # ------------------------------------------------------------------
    def get_settings(self) -> Dict:
        try:
            stored = self._db.get_setting(_DB_SETTINGS, {}) or {}
        except Exception:
            stored = {}
        s = dict(DEFAULT_SETTINGS)
        if isinstance(stored, dict):
            s.update(stored)
        # sanitize
        s['duration_minutes'] = max(0, int(s.get('duration_minutes', 5) or 0))
        s['potential_threshold_pct'] = max(
            1, min(100, int(s.get('potential_threshold_pct', 95) or 95)))
        s['max_exhaustion_pct'] = max(
            1, min(100, int(s.get('max_exhaustion_pct', 75) or 75)))
        try:
            s['queue_ttl_hours'] = max(0, min(720, int(s.get('queue_ttl_hours', 24) or 0)))
        except (TypeError, ValueError):
            s['queue_ttl_hours'] = 24
        # ⚡ CTR gate sanitize
        if str(s.get('ctr_gate_mode', 'off')).lower() not in ('off', 'anti_extreme', 'fresh_cross', 'both'):
            s['ctr_gate_mode'] = 'off'
        try:
            s['ctr_gate_stc_threshold'] = max(50, min(99, int(s.get('ctr_gate_stc_threshold', 75) or 75)))
        except (TypeError, ValueError):
            s['ctr_gate_stc_threshold'] = 75
        try:
            s['ctr_gate_lean_pct'] = max(0, min(100, int(s.get('ctr_gate_lean_pct', 50) or 50)))
        except (TypeError, ValueError):
            s['ctr_gate_lean_pct'] = 50
        try:
            s['ctr_gate_max_age_bars'] = max(0, min(50, int(s.get('ctr_gate_max_age_bars', 3) or 3)))
        except (TypeError, ValueError):
            s['ctr_gate_max_age_bars'] = 3
        s['queue1_enabled'] = bool(s.get('queue1_enabled', True))
        s['queue2_enabled'] = bool(s.get('queue2_enabled', False))
        # ── Queue 2 eject rules ──
        s['queue2_eject_ctr'] = bool(s.get('queue2_eject_ctr', False))
        s['queue2_eject_choch'] = bool(s.get('queue2_eject_choch', True))
        s['queue2_use_buttons'] = bool(s.get('queue2_use_buttons', False))
        s['queue2_use_btc'] = bool(s.get('queue2_use_btc', True))
        try:
            s['queue2_open_min_dec_score'] = max(0, min(100, int(s.get('queue2_open_min_dec_score', 20) or 0)))
        except (TypeError, ValueError):
            s['queue2_open_min_dec_score'] = 20
        s['queue2_hold_unknown_ctr'] = bool(s.get('queue2_hold_unknown_ctr', True))
        s['queue2_queue_all'] = bool(s.get('queue2_queue_all', False))
        s['queue2_reverse_via_queue'] = bool(s.get('queue2_reverse_via_queue', True))
        s['queue2_hold_and_wait'] = bool(s.get('queue2_hold_and_wait', True))
        try:
            s['queue2_ctr_neutral_pct'] = max(0, min(50, int(s.get('queue2_ctr_neutral_pct', 10) or 0)))
        except (TypeError, ValueError):
            s['queue2_ctr_neutral_pct'] = 10
        try:
            s['queue2_min_entry_score'] = max(0, min(100, int(s.get('queue2_min_entry_score', 45) or 0)))
        except (TypeError, ValueError):
            s['queue2_min_entry_score'] = 45
        try:
            s['queue2_open_min_entry_score'] = max(0, min(100, int(s.get('queue2_open_min_entry_score', 60) or 0)))
        except (TypeError, ValueError):
            s['queue2_open_min_entry_score'] = 60
        try:
            s['queue2_ttl_hours'] = max(0.0, min(72.0, float(s.get('queue2_ttl_hours', 6) or 0)))
        except (TypeError, ValueError):
            s['queue2_ttl_hours'] = 6
        s['ctr_mtf_enabled'] = bool(s.get('ctr_mtf_enabled', True))
        _valid_tfs = ('5m', '15m', '30m', '45m', '1h', '2h', '4h')
        _tfs = s.get('ctr_mtf_tfs') or ['5m', '15m', '45m', '1h', '4h']
        if isinstance(_tfs, str):
            _tfs = [t.strip().lower() for t in _tfs.split(',') if t.strip()]
        s['ctr_mtf_tfs'] = [t for t in _tfs if t in _valid_tfs] or ['5m', '15m', '45m', '1h', '4h']
        s['q2_auto_ob_sl'] = bool(s.get('q2_auto_ob_sl', False))
        try:
            s['queue2_eject_ctr_pct'] = max(0, min(100, int(s.get('queue2_eject_ctr_pct', 20) or 20)))
        except (TypeError, ValueError):
            s['queue2_eject_ctr_pct'] = 20
        try:
            s['q2_auto_ob_sl_buffer_pct'] = max(0.0, min(10.0, float(s.get('q2_auto_ob_sl_buffer_pct', 0.2) or 0.2)))
        except (TypeError, ValueError):
            s['q2_auto_ob_sl_buffer_pct'] = 0.2
        _tf = str(s.get('q2_auto_ob_sl_tf', '15m') or '15m').lower()
        s['q2_auto_ob_sl_tf'] = _tf if _tf in ('15m', '30m', '1h', '4h') else '15m'
        s['engine_smart_direction'] = bool(s.get('engine_smart_direction', False))
        s['use_potential_exit'] = bool(s.get('use_potential_exit', True))
        s['skip_wait_coins'] = bool(s.get('skip_wait_coins', False))
        s['enabled'] = bool(s.get('enabled', False))
        try:
            s['direction_smoothing_min'] = max(0, min(600,
                int(s.get('direction_smoothing_min', 45))))
        except (TypeError, ValueError):
            s['direction_smoothing_min'] = 45
        # Remove obsolete keys
        s.pop('mode', None)
        s.pop('max_positions', None)
        return s

    def update_settings(self, patch: Dict) -> Dict:
        s = self.get_settings()
        if isinstance(patch, dict):
            for k in DEFAULT_SETTINGS:
                if k in patch:
                    s[k] = patch[k]
            # The two engine-mode toggles are mutually exclusive: turning one ON
            # forces the other OFF. Both may be OFF (engine idle). When a patch
            # tries to enable both, the BTC mode wins by convention.
            if patch.get('start_engine_enabled') is True:
                s['start_engine_independent'] = False
            elif patch.get('start_engine_independent') is True:
                s['start_engine_enabled'] = False
            # Safety net: never persist both ON.
            if s.get('start_engine_enabled') and s.get('start_engine_independent'):
                s['start_engine_independent'] = False
        try:
            self._db.set_setting(_DB_SETTINGS, s)
        except Exception as e:
            print(f"[FuelFilter] settings persist error: {e}")
        # validated copy
        s2 = self.get_settings()
        if s2.get('enabled'):
            self.start()
        return s2

    def is_enabled(self) -> bool:
        return self.get_settings().get('enabled', False)

    def set_enabled(self, on: bool):
        self.update_settings({'enabled': bool(on)})
        if on:
            self.start()
            try:
                self._tick()
            except Exception as e:
                print(f"[FuelFilter] immediate tick error: {e}")

    # ------------------------------------------------------------------
    # ❤️ FF "база" — intercepted coins waiting to open
    # ------------------------------------------------------------------
    def intercept(self, symbol: str, side: str, kind: Optional[str] = None) -> bool:
        """Catch a fresh CHoCH/CHoCH+BOS signal and route it into the ENABLED
        queue(s). Direction = the signal side. `kind` = signal type ('choch' |
        'choch_bos') so a NEWER signal of the SAME direction but a DIFFERENT type
        replaces the stale queued one (and restarts its wait), instead of being
        treated as a duplicate of the old.

        Returns the ACTUAL disposition so the caller (and the chart marker) can
        tell the truth instead of a blanket «queued»:
          'queued'  — added to Q1 and/or Q2 (waiting).
          'dropped' — an enabled queue OWNED but REJECTED it (e.g. Q2 CTR gate).
                      NOT queued, NOT opened; the caller must NOT open directly.
          ''        — BOTH queues off (or invalid input) → the caller opens the
                      signal directly per its own flow.
        NOTE: when Queue-1 is OFF the signal is NEVER added to Queue-1; a disabled
        queue simply doesn't participate (only enabled queues can own/queue it).

        ⚡ Queue 2 hard CTR gate (at SIGNAL time): the signal is queued into Q2
        ONLY if the CTR lean already points the signal's way. If CTR ≠ side the
        signal is IGNORED entirely (not queued, never opened) — but still owned
        by FF, so it never falls through to a direct open."""
        sym = (symbol or '').upper().strip()
        side = (side or '').upper().strip()
        kind = (kind or '').lower().strip() or None
        if not sym or side not in ('LONG', 'SHORT'):
            return ''
        _kind_lbl = {'choch': 'CHoCH', 'choch_bos': 'CHoCH+BOS'}.get(kind, kind or '?')
        try:
            from detection.activity_log import log_activity
        except Exception:
            log_activity = lambda *a, **k: None
        s = self.get_settings()
        q1 = bool(s.get('queue1_enabled', True))
        q2 = bool(s.get('queue2_enabled', False))
        if not (q1 or q2):
            log_activity(sym, 'passthrough', 'Обидві черги вимкнені → сигнал іде у пряме відкриття', side=side, source='intercept')
            return ''   # both queues OFF → do NOT intercept; open directly
        now = time.time()
        opp = 'SHORT' if side == 'LONG' else 'LONG'
        # ⭐ PROFESSIONAL Q2 entry decision (hold-and-wait). A fresh CHoCH fires
        # AGAINST the CTR lean by nature, so we DON'T drop on opposite CTR — we
        # queue every QUALITY setup (ENTRY score ≥ min) and let the engine open it
        # once CTR pulls into alignment (reversal-pullback entry). ENTRY score is
        # the SETUP metric (signal type + OB confluence + CTR timing + fuel +
        # graded momentum), NOT the noisy hold-SCORE.
        _band = float(s.get('queue2_ctr_neutral_pct', 10) or 0)
        entry = self._entry_score_for(sym, side, kind) if q2 else None
        _ctr_state, _ctr_stc, _ctr_pct = (self._ctr_state(sym, _band) if q2
                                          else ('none', None, 0.0))
        _hold_and_wait = bool(s.get('queue2_hold_and_wait', True))
        _hold_unknown = bool(s.get('queue2_hold_unknown_ctr', True))
        _queue_all = bool(s.get('queue2_queue_all', False))
        _min_q = int(s.get('queue2_min_entry_score', 45) or 0)
        _entry_score = int(entry['score']) if entry else 0
        _quality_ok = bool(q2 and _entry_score >= _min_q)
        if q2 and _hold_and_wait:
            # Queue any quality signal regardless of CTR direction.
            q2_take = _quality_ok
        elif q2:
            # Legacy gate: also require CTR aligned / neutral / unknown at signal.
            _ctr_ok = (_ctr_state == side or _ctr_state == 'neutral'
                       or (_ctr_state == 'none' and _hold_unknown) or _queue_all)
            q2_take = bool(_quality_ok and _ctr_ok)
        else:
            q2_take = False
        # ⚡ Queue 2 opposite-CHoCH eject (default ON): the SAME coin arriving with
        # the OPPOSITE direction invalidates any waiting Q2 entry — drop it even if
        # THIS new signal isn't queued (its CTR gate may drop it, or it may go
        # straight to a trade). Applies whether or not q2_take (a take overwrites
        # the entry anyway; the eject matters when the new one isn't queued).
        eject_choch = bool(q2 and s.get('queue2_eject_choch', True))
        ejected_choch = False
        refreshed_q1 = refreshed_q2 = False   # same dir, NEWER type → replaced stale
        stale_removed_q2 = False
        changed = False

        def _is_stale(prev):
            # Same direction but a DIFFERENT signal TYPE → the queued one is stale.
            return bool(prev and prev.get('dir') == side
                        and prev.get('kind') not in (None, kind))

        with self._lock:
            if eject_choch:
                _cur2 = self._pending2.get(sym)
                if _cur2 and _cur2.get('dir') == opp:
                    self._pending2.pop(sym, None)
                    ejected_choch = True
                    changed = True
            if q1:
                prev = self._pending.get(sym) or {}
                refreshed_q1 = _is_stale(prev)
                # Keep the original wait ONLY for the very SAME signal (same dir
                # AND same type). A different type (CHoCH↔CHoCH+BOS) restarts it.
                _keep = (prev.get('dir') == side and prev.get('kind') == kind
                         and prev.get('added_at'))
                self._pending[sym] = {'dir': side, 'kind': kind,
                                      'added_at': prev.get('added_at') if _keep else now}
                changed = True
            if q2_take:
                prev2 = self._pending2.get(sym) or {}
                refreshed_q2 = _is_stale(prev2)
                _keep2 = (prev2.get('dir') == side and prev2.get('kind') == kind
                          and prev2.get('added_at'))
                self._pending2[sym] = {'dir': side, 'kind': kind,
                                       'added_at': prev2.get('added_at') if _keep2 else now,
                                       'entry_score': _entry_score,
                                       'ctr_signal': _ctr_state,
                                       'ctr_stc_signal': _ctr_stc}
                changed = True
            elif q2:
                # New same-dir signal of a DIFFERENT type that did NOT pass the
                # CTR gate → drop the stale queued entry (the newer read wins).
                if _is_stale(self._pending2.get(sym)):
                    self._pending2.pop(sym, None)
                    stale_removed_q2 = True
                    changed = True
            if changed:
                self._persist_state()
        if ejected_choch:
            self._engine_skip.pop(sym, None)
            print(f"[FF-Q2] видалено {sym}: протилежний CHoCH {side} (був у черзі {opp})")
            log_activity(sym, 'ejected', f'Черга-2: протилежний CHoCH {side} стер запис {opp}', side=side, source='Q2')
        # SCORE + CTR readings at this moment — appended to queued/dropped/eject
        # records so the 🧾 ДЕТАЛІ column shows WHY (per operator's request).
        _sc_ctr = self._score_ctr_detail(sym)
        _sc_ctr_sfx = f' | {_sc_ctr}' if _sc_ctr else ''
        if q1:
            _r1 = ' · новий тип замінив застарілий' if refreshed_q1 else ''
            log_activity(sym, 'queued', f'Черга-1 · {_kind_lbl}{_r1}{_sc_ctr_sfx}', side=side, source='Q1')
        # ENTRY-score + CTR-state suffix for Q2 records (the SETUP metric).
        def _ctr_words(state, stc, pct):
            if state == 'none':
                return 'CTR —'
            if state == 'neutral':
                return f'CTR нейтральний (STC {stc:.0f})'
            return f'CTR нахил {state} {pct:.0f}%'
        _entry_lbl = self._score_label_ua(entry['label']) if entry else '—'
        _q2_sfx = (f' | ENTRY {_entry_score} {_entry_lbl} | '
                   f'{_ctr_words(_ctr_state, _ctr_stc, _ctr_pct)}') if q2 else ''
        # 🔬 STRUCTURED fields (pivotable, no text-parsing) — stamped on Q2 records.
        _x = None
        if q2 and entry:
            _fd = self._fuel_dir_smoothed(sym) or {}
            try:
                _fstr = int(round(abs(float(_fd.get('dir') or 0)) * 100))
            except (TypeError, ValueError):
                _fstr = None
            _x = {
                'entry_score': _entry_score,
                'entry_label': entry.get('label'),
                'ctr_state': _ctr_state,
                'ctr_stc': _ctr_stc,
                'ctr_pct': round(_ctr_pct, 1),
                'fuel_dir': _fd.get('status'),
                'fuel_str': _fstr,
                'kind': kind,
                'price': _fd.get('mark_price'),
                'comp': entry.get('components'),
            }
            # Record the MATURE Decision Center verdict alongside — for the data-
            # driven consolidation of the two entry models (see _decision_compact).
            _dec = self._decision_compact(sym, _fd.get('mark_price'), side)
            if _dec:
                _x['dec'] = _dec
            # Multi-TF CTR confluence (record-only) — trend vs timing across TFs.
            _mtf = self._ctr_confluence(sym, side)
            if _mtf:
                _x['ctr_mtf'] = _mtf
        if q2_take:
            _r2 = ' · новий тип замінив застарілий' if refreshed_q2 else ''
            if _ctr_state == side:
                _why = 'CTR у бік сигналу'
            elif _ctr_state == 'neutral':
                _why = 'CTR нейтральний — чекаємо підтягування'
            elif _ctr_state == 'none':
                _why = 'CTR ще не порахований — чекаємо'
            else:   # opposite → held (hold-and-wait), waiting for CTR pullback
                _why = 'CTR поки проти — тримаємо до розвороту CTR (hold&wait)'
            log_activity(sym, 'queued', f'Черга-2 · {_kind_lbl} ({_why}){_r2}{_q2_sfx}', side=side, source='Q2', extra=_x)
        elif q2 and not q2_take:
            # Not queued → quality too low (hold&wait) or CTR gate (legacy).
            if _entry_score < _min_q:
                _drx = f'ENTRY {_entry_score} < мін {_min_q} — сетап заслабкий'
            else:
                _drx = f'CTR {_ctr_state} ≠ {side}'
            print(f"[FF-Q2] відхилено {sym} {side}: {_drx}")
            if stale_removed_q2:
                log_activity(sym, 'ejected', f'Черга-2: застарілий {_kind_lbl} знято — {_drx}{_q2_sfx}', side=side, source='Q2', extra=_x)
            else:
                log_activity(sym, 'dropped', f'Черга-2 · {_kind_lbl}: {_drx} — відхилено{_q2_sfx}', side=side, source='Q2', extra=_x)
        print(f"[FuelFilter] intercepted {sym} {side} → Q1={q1} Q2={'take' if q2_take else ('drop' if q2 else 'off')}")
        # Return the ACTUAL disposition so the caller (and the chart marker) tell
        # the truth: 'queued' — added to a queue; 'dropped' — an enabled queue
        # OWNED it but rejected it (e.g. Q2 CTR gate) → NOT queued, not opened.
        if q1 or q2_take:
            return 'queued'
        return 'dropped'   # q2 enabled but the signal didn't pass its gate

    def queue2_on_choch(self, symbol: str, direction: str) -> bool:
        """Called by the SMC scanner for EVERY fresh CHoCH (same source that
        draws the chart) so a Queue-2 coin TRACKS the opposite CHoCH directly —
        not only when an opposite signal happens to pass the intercept pipeline
        (which HTF/dedup/button filters can swallow).

        If Queue-2 'eject on opposite CHoCH' is ON and the coin waits in Queue 2
        with the OPPOSITE direction, it is dropped — the setup that queued it is
        invalidated by a fresh counter-CHoCH on the chart.
        `direction`: 'bull'/'bear' (scanner) or 'LONG'/'SHORT'."""
        sym = (symbol or '').upper().strip()
        d = (direction or '').lower()
        choch_side = ('LONG' if d in ('bull', 'long')
                      else ('SHORT' if d in ('bear', 'short') else None))
        if not sym or choch_side is None:
            return False
        s = self.get_settings()
        if not s.get('queue2_enabled') or not s.get('queue2_eject_choch', True):
            return False
        opp = 'SHORT' if choch_side == 'LONG' else 'LONG'
        removed = False
        with self._lock:
            cur = self._pending2.get(sym)
            if cur and cur.get('dir') == opp:
                self._pending2.pop(sym, None)
                self._persist_state()
                removed = True
        if removed:
            self._engine_skip.pop(sym, None)
            print(f"[FF-Q2] видалено {sym}: протилежний CHoCH {choch_side} на графіку (був у черзі {opp})")
            try:
                from detection.activity_log import log_activity
                log_activity(sym, 'ejected', f'Черга-2: протилежний CHoCH {choch_side} на графіку (був {opp})', side=opp, source='Q2')
            except Exception:
                pass
        return removed

    def remove_pending(self, symbol: str) -> bool:
        """Drop a coin from the waiting base (user ✕)."""
        if not _q_allowed(7):   # OP 7: remove_pending
            return False
        sym = (symbol or '').upper().strip()
        with self._lock:
            existed = self._pending.pop(sym, None) is not None
            if existed:
                self._persist_state()
        return existed

    def clear_pending(self) -> int:
        if not _q_allowed(8):   # OP 8: clear_pending
            return 0
        with self._lock:
            n = len(self._pending)
            self._pending = {}
            self._persist_state()
        return n

    def _entry_gates(self) -> tuple:
        """(allow_long, allow_short) from TM's main directional buttons — used
        both to FILTER the FF table display and to select open candidates."""
        try:
            tm = self._get_tm() if self._get_tm else None
            ts = tm.get_settings() if tm and hasattr(tm, 'get_settings') else {}
            return (bool(ts.get('allow_long_entries', True)),
                    bool(ts.get('allow_short_entries', True)))
        except Exception:
            return (True, True)

    # ------------------------------------------------------------------
    # scan-list (whitelist) — which WATCHLIST coins FF is allowed to scan.
    # Lets the operator pick a handful of coins instead of hammering every
    # coin on the board (load control). Empty list = scan NOTHING (opt-in).
    # ------------------------------------------------------------------
    def get_scan_list(self) -> List[str]:
        try:
            lst = self._db.get_setting(_DB_SCAN_LIST, []) or []
        except Exception:
            lst = []
        if not isinstance(lst, list):
            return []
        return [str(s).upper() for s in lst]

    def set_scan(self, symbol: str, on: bool) -> List[str]:
        """Add/remove a symbol from the scan-list. Returns the new list."""
        sym = (symbol or '').upper().strip()
        if not sym:
            return self.get_scan_list()
        lst = self.get_scan_list()
        if on:
            if sym not in lst:
                lst.append(sym)
        else:
            lst = [s for s in lst if s != sym]
            # No longer scanning → drop any pending timer for it (an open
            # position stays managed until it exits on its own rules).
            self._timers.pop(sym, None)
        try:
            self._db.set_setting(_DB_SCAN_LIST, lst)
        except Exception as e:
            print(f"[FuelFilter] scan-list persist error: {e}")
        return lst

    def is_scanned(self, symbol: str) -> bool:
        return (symbol or '').upper() in self.get_scan_list()

    # ------------------------------------------------------------------
    # state persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        try:
            st = self._db.get_setting(_DB_STATE, {}) or {}
        except Exception:
            st = {}
        if isinstance(st, dict):
            self._fuel_managed = st.get('fuel_managed', {}) or {}
            # Timers belong to ACTIVE managed positions only. Drop any orphan
            # timer left over from an old strategy / previous session whose coin
            # is no longer a tracked position — otherwise the table shows ghost
            # rows for coins that never fired a fresh signal this session.
            _timers_raw = st.get('timers', {}) or {}
            self._timers = {s: v for s, v in _timers_raw.items()
                            if s in self._fuel_managed}
            # Anomalies: coins that held fuel longer than anomaly_hours. They
            # live in their OWN table, persist across fuel loss, and are removed
            # only by the user (manual delete / clear). {symbol: {...}}
            self._anomalies = st.get('anomalies', {}) or {}
            # Persistent per-coin funding archive (separate DB key — can grow).
            try:
                fa = self._db.get_setting(_DB_FUNDING_ARCHIVE, {}) or {}
                if isinstance(fa, dict):
                    self._funding_archive = {str(k).upper(): (v or [])
                                             for k, v in fa.items()
                                             if isinstance(v, list)}
                    print(f"[FuelFilter] restored funding archive: "
                          f"{len(self._funding_archive)} coin(s)")
            except Exception as e:
                print(f"[FuelFilter] funding archive load error: {e}")
            # Muted funding coins (drop already-expired ones).
            try:
                _now = time.time()
                fm = st.get('funding_muted', {}) or {}
                self._funding_muted = {str(k).upper(): float(v)
                                       for k, v in fm.items()
                                       if float(v) > _now}
            except Exception:
                self._funding_muted = {}
            # Candle-confirm attempt counters per coin — restored so the
            # "🕯️ Спроби" column and the "Opened by" attempt number survive
            # a bot restart instead of resetting to 0.
            ea = st.get('engine_attempts', {}) or {}
            if isinstance(ea, dict):
                self._engine_attempts = {str(k).upper(): int(v)
                                         for k, v in ea.items()
                                         if str(v).lstrip('-').isdigit()}
            # Direction-smoothing state.
            em = st.get('fuel_ema', {}) or {}
            if isinstance(em, dict):
                self._fuel_ema = {str(k).upper(): float(v) for k, v in em.items()}
            hy = st.get('fuel_hyst', {}) or {}
            if isinstance(hy, dict):
                self._fuel_hyst = {str(k).upper(): (v if v in ('LONG', 'SHORT') else None)
                                   for k, v in hy.items()}
            # Restore the BTC ММ SESSION first (direction + start time).
            bvd = st.get('btc_verdict_dir')
            self._btc_verdict_dir = bvd if bvd in ('LONG', 'SHORT') else None
            try:
                self._btc_verdict_since = float(st.get('btc_verdict_since') or 0.0)
            except (TypeError, ValueError):
                self._btc_verdict_since = 0.0
            self._btc_paused = False
            # Restore the entry queue PER SESSION. We persist the queue now and
            # bring it back on boot, tied to the session it belonged to. The
            # session-flip logic in _update_btc_verdict handles staleness: on the
            # first tick, if the live ММ is OPPOSITE the restored session, the
            # queue is cleared (session mismatch → fresh start). If the ML is the
            # same direction (or WAIT/pause), the queued coins are still valid and
            # keep waiting. This replaces the old blanket-ephemeral behaviour.
            pend = st.get('pending', {}) or {}
            if isinstance(pend, dict):
                self._pending = {str(k).upper(): v for k, v in pend.items()
                                 if isinstance(v, dict) and v.get('dir') in ('LONG', 'SHORT')}
            pend2 = st.get('pending2', {}) or {}
            if isinstance(pend2, dict):
                self._pending2 = {str(k).upper(): v for k, v in pend2.items()
                                  if isinstance(v, dict) and v.get('dir') in ('LONG', 'SHORT')}
            if (self._fuel_managed or self._anomalies or self._engine_attempts
                    or self._timers or self._pending):
                print(f"[FuelFilter] restored {len(self._pending)} queued "
                      f"(session={self._btc_verdict_dir or '—'}), "
                      f"{len(self._fuel_managed)} tracked position(s), "
                      f"{len(self._timers)} timer(s), "
                      f"{len(self._anomalies)} anomaly(ies), "
                      f"{len(self._engine_attempts)} attempt-counter(s) from DB")

    def _persist_state(self):
        try:
            self._db.set_setting(_DB_STATE, {
                'timers': self._timers,
                'fuel_managed': self._fuel_managed,
                'anomalies': self._anomalies,
                'engine_attempts': self._engine_attempts,
                'fuel_ema': self._fuel_ema,
                'fuel_hyst': self._fuel_hyst,
                'btc_verdict_dir': self._btc_verdict_dir,
                'btc_verdict_since': self._btc_verdict_since,
                'pending': self._pending,
                'pending2': self._pending2,
                'funding_muted': self._funding_muted,
            })
        except Exception as e:
            print(f"[FuelFilter] state persist error: {e}")

    # ------------------------------------------------------------------
    # per-coin funding ARCHIVE (history of what happened & when)
    # ------------------------------------------------------------------
    def _push_funding_arch(self, sym, event, snap, now):
        """Append one archive record for `sym`. Re-appearing coins extend the
        SAME series (self._funding_archive[sym] already exists)."""
        rec = {
            't': int(now), 'ev': event,
            'mm': snap.get('mm'), 'mm_dir': snap.get('dir'),
            'funding': snap.get('funding'), 'price': snap.get('price'),
            'vol24h': snap.get('vol24h'),
            'btc_dir': self._btc_verdict_dir,
            'btc_paused': bool(self._btc_paused),
        }
        arr = self._funding_archive.get(sym)
        if arr is None:
            # cap distinct coins — drop the least-recently-updated archive
            if len(self._funding_archive) >= FUNDING_ARCH_COINS_MAX:
                try:
                    oldest = min(self._funding_archive,
                                 key=lambda k: (self._funding_archive[k][-1]['t']
                                                if self._funding_archive[k] else 0))
                    self._funding_archive.pop(oldest, None)
                except Exception:
                    pass
            arr = self._funding_archive[sym] = []
        arr.append(rec)
        if len(arr) > FUNDING_ARCH_MAX:
            del arr[:len(arr) - FUNDING_ARCH_MAX]
        self._funding_arch_dirty = True

    def _archive_funding_pass(self, now):
        """Central event detector for the funding table → archive. Compares the
        current holding coins to the previous tick and records appear / flip /
        exit / periodic sample. One place → no edits to the scan branches."""
        prev = self._funding_prev_state
        cur = {}
        for sym, a in self._anomalies.items():
            if a.get('funding') and a.get('holding'):
                rate = self._funding_rates.get(sym)
                if rate is None:
                    rate = a.get('rate')
                cur[sym] = {
                    'dir': a.get('dir'), 'mm': a.get('mm_str'),
                    'price': a.get('last_price') or a.get('start_price'),
                    'funding': rate, 'vol24h': a.get('vol24h'),
                }
        for sym in (set(prev) | set(cur)):
            ps, cs = prev.get(sym), cur.get(sym)
            if cs and not ps:
                self._push_funding_arch(sym, 'appear', cs, now)
                self._funding_arch_last[sym] = now
            elif cs and ps and cs.get('dir') != ps.get('dir'):
                self._push_funding_arch(sym, 'flip', cs, now)
                self._funding_arch_last[sym] = now
            elif ps and not cs:
                self._push_funding_arch(sym, 'exit', ps, now)
            elif cs and ps:
                if (now - self._funding_arch_last.get(sym, 0)) >= FUNDING_ARCH_SAMPLE_SEC:
                    self._push_funding_arch(sym, 'sample', cs, now)
                    self._funding_arch_last[sym] = now
        self._funding_prev_state = cur
        # Throttled persist of the (growing) archive to its own DB key.
        if self._funding_arch_dirty and (now - self._funding_arch_saved_at) >= 30:
            try:
                self._db.set_setting(_DB_FUNDING_ARCHIVE, self._funding_archive)
                self._funding_arch_saved_at = now
                self._funding_arch_dirty = False
            except Exception as e:
                print(f"[FuelFilter] funding archive persist error: {e}")

    def get_funding_history(self, sym: str) -> List[Dict]:
        """Full archived series for one funding coin (may span several
        appear→exit episodes)."""
        sym = (sym or '').upper()
        with self._lock:
            live = bool(sym in self._anomalies and self._anomalies[sym].get('holding'))
            return {'ok': True, 'symbol': sym, 'live': live,
                    'history': list(self._funding_archive.get(sym) or [])}

    def delete_funding_archive(self, sym: str) -> bool:
        """Remove one coin's archive."""
        sym = (sym or '').upper()
        with self._lock:
            existed = self._funding_archive.pop(sym, None) is not None
            self._funding_prev_state.pop(sym, None)
            self._funding_arch_last.pop(sym, None)
            if existed:
                try:
                    self._db.set_setting(_DB_FUNDING_ARCHIVE, self._funding_archive)
                except Exception as e:
                    print(f"[FuelFilter] archive delete persist: {e}")
        return existed

    def clear_funding_archive(self) -> int:
        """Wipe the whole funding archive."""
        with self._lock:
            n = len(self._funding_archive)
            self._funding_archive = {}
            self._funding_prev_state = {}
            self._funding_arch_last = {}
            try:
                self._db.set_setting(_DB_FUNDING_ARCHIVE, {})
            except Exception as e:
                print(f"[FuelFilter] archive clear persist: {e}")
        return n

    def mute_funding(self, sym: str, hours: float = 24.0) -> float:
        """Mute a funding coin for `hours` — it's ignored entirely (no row, no
        TG, no archive) until then. Removes it from the table immediately.
        Returns the until-timestamp."""
        sym = (sym or '').upper()
        try:
            hours = max(0.0, min(720.0, float(hours)))
        except (TypeError, ValueError):
            hours = 24.0
        until = time.time() + hours * 3600.0
        with self._lock:
            self._funding_muted[sym] = until
            self._anomalies.pop(sym, None)
            self._funding_notify_at.pop(sym, None)
            self._funding_prev_state.pop(sym, None)
            self._persist_state()
        return until

    def unmute_funding(self, sym: str) -> bool:
        sym = (sym or '').upper()
        with self._lock:
            existed = self._funding_muted.pop(sym, None) is not None
            if existed:
                self._persist_state()
        return existed

    def list_funding_muted(self) -> List[Dict]:
        """Active mutes (auto-drops expired), with remaining seconds."""
        now = time.time()
        with self._lock:
            for s in [k for k, v in self._funding_muted.items() if v <= now]:
                self._funding_muted.pop(s, None)
            return [{'symbol': s, 'until': v, 'remaining_sec': int(v - now)}
                    for s, v in sorted(self._funding_muted.items(),
                                       key=lambda kv: kv[1])]

    def _is_funding_muted(self, sym: str, now: float) -> bool:
        u = self._funding_muted.get(sym)
        if u is None:
            return False
        if u <= now:
            self._funding_muted.pop(sym, None)
            return False
        return True

    def get_funding_archive_list(self) -> List[Dict]:
        """Index of archived coins (incl. ones no longer on the radar), newest
        activity first."""
        with self._lock:
            out = []
            for s, arr in self._funding_archive.items():
                if not arr:
                    continue
                live = bool(s in self._anomalies and self._anomalies[s].get('holding'))
                out.append({'symbol': s, 'count': len(arr),
                            'first_t': arr[0].get('t'), 'last_t': arr[-1].get('t'),
                            'live': live})
            out.sort(key=lambda x: -(x.get('last_t') or 0))
            return out

    # ------------------------------------------------------------------
    # per-coin SCORE for the active-timers table
    # ------------------------------------------------------------------
    @staticmethod
    def _score_label(score: float):
        """Map a 0..100 hold-score to a (label, color) verdict. The label stays
        ENGLISH — it's an internal key compared across the code (== 'STRONG
        HOLD'); use _score_label_ua() to display it in Ukrainian."""
        if score >= 72:
            return ('STRONG HOLD', '#16a34a')   # green
        if score >= 55:
            return ('HOLD', '#84cc16')           # lime
        if score >= 40:
            return ('NEUTRAL', '#eab308')        # amber
        if score >= 25:
            return ('WEAK', '#f97316')           # orange
        return ('EXHAUSTED', '#ef4444')          # red

    # SCORE label EN→UA for DISPLAY only (internal keys stay English for logic).
    _SCORE_LABEL_UA = {
        'STRONG HOLD': 'СИЛЬНЕ УТРИМАННЯ',
        'HOLD': 'УТРИМАННЯ',
        'NEUTRAL': 'НЕЙТРАЛЬНО',
        'WEAK': 'СЛАБКЕ',
        'EXHAUSTED': 'ВИСНАЖЕНО',
    }

    @classmethod
    def _score_label_ua(cls, label):
        return cls._SCORE_LABEL_UA.get(label, label or '—')

    def _candle_momentum(self, symbol, tf='5m'):
        """LIVE price direction from the last 2 closed candles (close vs open).
        Returns (dir, strength): dir ∈ {'LONG','SHORT',None}, strength ∈ {0,1}.
        Both bars must agree (15m × 2 = ~30-min impulse); a split → neutral.
        Uses the cached kline helper (no extra exchange load on a warm cache)."""
        try:
            kl = self._candle_klines(symbol, tf)
        except Exception:
            kl = None
        if not kl or len(kl) < 3:
            return (None, 0.0)
        last2 = kl[:-1][-2:]
        if len(last2) < 2:
            return (None, 0.0)
        ups = sum(1 for k in last2 if float(k.get('p', 0)) > float(k.get('o', 0)))
        downs = sum(1 for k in last2 if float(k.get('p', 0)) < float(k.get('o', 0)))
        if ups == 2:
            return ('LONG', 1.0)
        if downs == 2:
            return ('SHORT', 1.0)
        return (None, 0.0)

    def _candle_momentum_graded(self, symbol, tf='5m', bars=3):
        """GRADED price momentum (replaces the binary 2-candle vote in SCORE).
        Signed net move over the last `bars` CLOSED candles, normalised by ATR so
        it is comparable across coins, mapped to 0..1 magnitude. Returns
        (dir, strength): dir ∈ {'LONG','SHORT',None}, strength 0..1 that SCALES
        with the real size of the move — a big impulse scores far higher than a
        pair of dojis (the old binary flaw). (None, 0.0) when data is thin."""
        try:
            kl = self._candle_klines(symbol, tf)
        except Exception:
            kl = None
        if not kl or len(kl) < bars + 2:
            return (None, 0.0)
        closed = kl[:-1]                      # drop the still-forming bar
        seg = closed[-bars:]
        try:
            net = float(seg[-1].get('p', 0)) - float(seg[0].get('o', 0))
        except (TypeError, ValueError):
            return (None, 0.0)
        # ATR over up to the last 14 closed bars (Wilder-ish simple mean of TR).
        trs = []
        for i in range(1, len(closed)):
            try:
                h = float(closed[i].get('h', 0)); l = float(closed[i].get('l', 0))
                pc = float(closed[i - 1].get('p', 0))
            except (TypeError, ValueError):
                continue
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        trs = trs[-14:]
        atr = (sum(trs) / len(trs)) if trs else 0.0
        if atr <= 0:
            return (None, 0.0)
        # Normalise: a move of ~1×(ATR·√bars) → strength ≈ 1.0 (full momentum).
        z = net / (atr * (bars ** 0.5))
        strength = max(0.0, min(1.0, abs(z)))
        d = 'LONG' if net > 0 else ('SHORT' if net < 0 else None)
        return (d, strength)

    def _ctr_state(self, symbol: str, band_pct: float = 0.0):
        """CTR reading as (state, stc, pct) with a NEUTRAL dead-zone:
          state ∈ 'none'    — no CTR data (STC is None → truly unknown),
                  'neutral' — STC present but |lean| < band_pct (incl. exactly 50),
                  'LONG'/'SHORT' — a real lean of ≥ band_pct.
        Fixes the old bug where STC==50 was reported as «CTR невідомий»: 50 is a
        computed NEUTRAL value, not missing data. pct = |STC−50|/50·100."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            stc = ((fe.get(symbol) or {}).get('ctr') or {}).get('stc') if fe else None
        except Exception:
            stc = None
        if stc is None:
            return ('none', None, 0.0)
        try:
            stc = float(stc)
        except (TypeError, ValueError):
            return ('none', None, 0.0)
        pct = abs(stc - 50.0) / 50.0 * 100.0
        try:
            band = float(band_pct or 0)
        except (TypeError, ValueError):
            band = 0.0
        if pct < band:
            return ('neutral', stc, pct)
        if stc > 50:
            return ('SHORT', stc, pct)
        if stc < 50:
            return ('LONG', stc, pct)
        return ('neutral', stc, pct)

    def _ctr_confluence(self, symbol: str, side: Optional[str] = None) -> Optional[Dict]:
        """⚡ Multi-timeframe CTR confluence — RECORD-ONLY (does not gate yet).
        Reads STC on each configured TF and splits the picture into TREND context
        (1h/4h) vs ENTRY timing (5m/15m), with the same neutral dead-zone as the
        single-TF gate. Returns a compact dict for the 🧾 log + chronology so we
        can later test whether multi-TF alignment predicts PnL better than the
        lone 15m CTR (then optionally gate on it).

        NOTE(calibration — DO WHEN DATA READY): validate `align` vs realised PnL;
        if it beats the single-TF CTR, wire it into the Queue-2 open gate. The
        trend/timing split and weights below are a PROVISIONAL first cut."""
        try:
            s = self.get_settings()
            if not s.get('ctr_mtf_enabled', True):
                return None
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe or not hasattr(fe, 'get_ctr_multi_tf'):
                return None
            tfs = list(s.get('ctr_mtf_tfs') or ['5m', '15m', '45m', '1h', '4h'])
            band = float(s.get('queue2_ctr_neutral_pct', 10) or 0)
            m = fe.get_ctr_multi_tf(symbol, tfs)
        except Exception:
            return None

        def _st(stc):
            if stc is None:
                return ('none', 0.0)
            pct = abs(float(stc) - 50.0) / 50.0 * 100.0
            if pct < band:
                return ('neutral', pct)
            return (('SHORT', pct) if stc > 50 else ('LONG', pct))

        TREND_TFS = {'1h', '2h', '4h'}
        TIMING_TFS = {'5m', '15m'}
        per, trend_votes, timing_votes = {}, [], []
        for tf in tfs:
            stc = (m.get(tf) or {}).get('stc')
            st, pct = _st(stc)
            per[tf] = {'stc': stc, 'state': st, 'pct': round(pct, 1)}
            if st in ('LONG', 'SHORT'):
                if tf in TREND_TFS:
                    trend_votes.append(st)
                if tf in TIMING_TFS:
                    timing_votes.append(st)

        def _majority(votes):
            if not votes:
                return None
            l, sh = votes.count('LONG'), votes.count('SHORT')
            return 'LONG' if l > sh else ('SHORT' if sh > l else None)

        trend = _majority(trend_votes)
        timing = _majority(timing_votes)
        align = None
        if side in ('LONG', 'SHORT'):
            score = 0.0
            score += 60 if trend == side else (30 if trend is None else 0)
            score += 40 if timing == side else (20 if timing is None else 0)
            align = int(round(score))
        return {'per_tf': per, 'trend': trend, 'timing': timing,
                'align': align, 'side': side}

    def _decision_compact(self, symbol: str, price, side=None) -> Optional[Dict]:
        """Compact Decision Center snapshot (the MATURE evaluate_entry model in
        trade_manager) for recording ALONGSIDE the FF ENTRY score, so we can
        later prove on data which predicts PnL better and consolidate.

        NOTE(calibration): once ≥ ~50 closed trades are collected, compare
        `ff_entry_score` vs `dec.score`→PnL correlation; gate Queue-2 on whichever
        wins, then retire the loser. Do NOT swap the live per-tick gate before
        that evidence — evaluate_entry is heavier and its weights are unfitted."""
        try:
            tm = self._get_tm() if self._get_tm else None
            if not tm or not hasattr(tm, 'compute_decision') or not price:
                return None
            dec = tm.compute_decision(symbol, float(price))
            if not dec:
                return None
            _score = (dec.get('long_score') if side == 'LONG'
                      else dec.get('short_score') if side == 'SHORT' else None)
            return {'reco': dec.get('recommended'), 'verdict': dec.get('verdict'),
                    'prob_long': dec.get('prob_long'),
                    'prob_short': dec.get('prob_short'),
                    'score': _score,
                    'long_score': dec.get('long_score'),
                    'short_score': dec.get('short_score')}
        except Exception:
            return None

    def _entry_score_for(self, symbol: str, side: str, kind=None) -> Dict:
        """⭐ ENTRY score (0..100) — SETUP quality for a candidate entry, evaluated
        STRICTLY in the SIGNAL direction (`side`). Separate from `_timer_score_for`
        (which is HOLD quality of an already-open trade and lets price momentum
        hijack the direction). Deterministic, professional weighting:

          • signal (15%)   — CHoCH+BOS (confirmed) > bare CHoCH.
          • ob     (25%)   — «Require OB Match» OB agrees with `side` (fresh CHoCH-
                             created OB scores full; BOS-created less; none/opposite 0).
          • ctr    (20%)   — CTR TIMING: lean in `side` scores by strength; NEUTRAL
                             (dead-zone) is half credit (fine for a pullback entry);
                             opposite lean scores 0 (but does NOT drop — hold&wait).
          • fuel   (20%)   — liq-fuel imbalance aligned with `side`.
          • mom    (20%)   — GRADED price momentum aligned with `side`.

        Returns {score,label,color,dir(=side),components}."""
        side = (side or '').upper()
        s = self.get_settings()
        comp = {}
        # 1) Signal type.
        # NOTE(calibration — DO WHEN DATA READY): the CHoCH+BOS > CHoCH ranking
        # (1.0 vs 0.7) is an ASSUMPTION. When the monitoring + chronology sample
        # is ready, SPLIT the closed trades by `ff_kind` (choch vs choch_bos) —
        # both are exported per trade — and compare win-rate / avg PnL / MAE /
        # MFE / time-to-peak. Determine which signal type is MAXIMALLY profitable
        # (and whether the edge differs by direction / CTR-mtf alignment), then
        # RE-SET these weights (or gate/queue only the winning type) from data.
        k = (kind or '').lower()
        comp['signal'] = 1.0 if k == 'choch_bos' else (0.7 if k == 'choch' else 0.5)
        # 2) OB confluence (same «Require OB Match» row the gate/auto-SL use).
        ob_c = 0.0
        try:
            from storage.db_operations import get_db
            ob_tf = '1h'
            tm = self._get_tm() if self._get_tm else None
            scanner = getattr(tm, 'scanner', None) if tm else None
            if scanner is not None:
                ob_tf = scanner._settings.get('ob_filter_timeframe', '1h')
            row = get_db().get_smc_ob_state(symbol, ob_tf)
            if row and row.get('bias'):
                wanted = 'BULLISH' if side == 'LONG' else 'BEARISH'
                if row.get('bias') == wanted:
                    ob_c = 1.0 if (row.get('created_by_tag') or '').upper() == 'CHOCH' else 0.6
        except Exception:
            ob_c = 0.0
        comp['ob'] = ob_c
        # 3) CTR timing (with neutral dead-zone).
        band = float(s.get('queue2_ctr_neutral_pct', 10) or 0)
        state, stc, pct = self._ctr_state(symbol, band)
        if state == side:
            comp['ctr'] = max(0.4, min(1.0, pct / 100.0))
        elif state in ('neutral', 'none'):
            comp['ctr'] = 0.5
        else:
            comp['ctr'] = 0.0        # opposite lean → 0 credit (still queued)
        # 4) Fuel aligned with side.
        fmag = 0.0
        try:
            fd = self._fuel_dir_smoothed(symbol)
            signed = float(fd['dir']) if fd and fd.get('dir') is not None else None
            if signed is not None:
                aligned = signed if side == 'LONG' else -signed
                fmag = max(0.0, min(1.0, aligned / 0.35))
        except Exception:
            fmag = 0.0
        comp['fuel'] = fmag
        # 5) Graded momentum aligned with side.
        mom = 0.0
        try:
            tf = s.get('engine_candle_tf', '5m')
            mdir, mstr = self._candle_momentum_graded(symbol, tf)
            if mdir == side:
                mom = mstr
        except Exception:
            mom = 0.0
        comp['mom'] = mom
        score = 100.0 * (0.15 * comp['signal'] + 0.25 * comp['ob']
                         + 0.20 * comp['ctr'] + 0.20 * comp['fuel']
                         + 0.20 * comp['mom'])
        score = int(round(max(0.0, min(100.0, score))))
        label, color = self._score_label(score)
        return {'score': score, 'label': label, 'color': color,
                'dir': side, 'components': comp,
                'ctr_state': state, 'ctr_stc': stc, 'ctr_pct': round(pct, 1)}

    def _timer_score_for(self, symbol, direction, held_sec, exhaustion,
                         dur_sec, tf='5m'):
        """Per-coin hold quality + its OWN live direction.

        CRITICAL: the displayed direction follows ACTUAL PRICE ACTION (recent
        candles), NOT just the liq-fuel bias. Fuel tells where liquidity is
        stacked; price tells where the coin is going RIGHT NOW. If a coin is
        dumping while fuel is long-biased, the SCORE must say SHORT (and flag a
        conflict) — not 'STRONG HOLD 🟢'. Direction priority:
            price momentum → fuel status → timer direction.

        Magnitude blends, all relative to that live direction:
          • room     (30%) — 1 − exhaustion (how much of the move is left)
          • hold     (15%) — ММ hold duration vs the show threshold (~3× sat.)
          • fuel     (25%) — liq-fuel imbalance aligned with the direction
          • momentum (30%) — candle momentum aligned with the direction
        When fuel bias and price momentum DISAGREE the score is hard-capped
        (→ WEAK at best) and `conflict` is set, because the ММ setup is being
        violated by price. Returns {score,label,color,dir,conflict}."""
        # Fuel bias (where liquidity sits) — SMOOTHED + hysteresis, read-only.
        fuel_dir = None
        signed = None
        try:
            fd = self._fuel_dir_smoothed(symbol)
            if fd:
                if fd.get('status') in ('LONG', 'SHORT'):
                    fuel_dir = fd['status']
                if fd.get('dir') is not None:
                    signed = float(fd['dir'])
        except Exception:
            pass

        # Price momentum (what the coin is actually doing now).
        # GRADED momentum (magnitude-scaled, not the old binary 2-candle vote).
        # A tiny/mixed move (< 0.25) is treated as NO clear momentum so noise
        # can't set the displayed direction — direction then falls to fuel/timer.
        price_dir, pstrength = self._candle_momentum_graded(symbol, tf)
        if pstrength < 0.25:
            price_dir = None

        # Displayed direction: price action wins, then fuel, then timer.
        live_dir = price_dir or fuel_dir or direction
        conflict = bool(fuel_dir and price_dir and fuel_dir != price_dir)

        # Exhaustion (room) for the live direction. Compute it whenever it
        # wasn't supplied (the queue passes None) OR when the live direction
        # differs from the signal — otherwise `exf` stayed None and the
        # "Виснаженість" column always showed "—".
        ex = exhaustion
        if ex is None or live_dir != direction:
            try:
                ex = self._exhaustion(symbol, live_dir)
            except Exception:
                ex = exhaustion
        try:
            exf = float(ex) if ex is not None else None
        except (TypeError, ValueError):
            exf = None
        room = 0.5 if exf is None else max(0.0, min(1.0, (100.0 - exf) / 100.0))

        # Hold conviction.
        ratio = (held_sec / dur_sec) if dur_sec and dur_sec > 0 else 1.0
        hold = max(0.0, min(1.0, ratio / 3.0))

        # Fuel magnitude aligned with the live direction (against → 0).
        fmag = 0.0
        if signed is not None:
            aligned = signed if live_dir == 'LONG' else -signed
            # /0.35 (was /0.5): the liq-fuel imbalance rarely reaches 0.5, so
            # the harsher divisor kept this term tiny and dragged SCORE down.
            fmag = max(0.0, min(1.0, aligned / 0.35))

        # Momentum aligned with the live direction.
        mom = pstrength if (price_dir and price_dir == live_dir) else 0.0

        # Weights. For a QUEUED coin (held_sec == 0) the 'hold' term is not
        # meaningful yet — it would otherwise sit at 0 and drag every ❤️ queue
        # SCORE down into WEAK/EXHAUSTED (capping the max at ~85). Redistribute
        # its weight across the live components so the queue SCORE uses the full
        # range and stays sensitive. Open positions (held>0) keep all four.
        w_room, w_hold, w_fuel, w_mom = 0.30, 0.15, 0.25, 0.30
        if not held_sec or held_sec <= 0:
            _tw = w_room + w_fuel + w_mom
            w_room, w_fuel, w_mom, w_hold = w_room / _tw, w_fuel / _tw, w_mom / _tw, 0.0
        score = 100.0 * (w_room * room + w_hold * hold + w_fuel * fmag + w_mom * mom)

        # Conflict: price is fighting the fuel setup → not a healthy hold.
        if conflict:
            score = min(score, 38)        # → WEAK at best
        # Exhaustion override (FF exits on exhaustion).
        if exf is not None:
            # Exhaustion already drives `room` (30% weight) — keep only a single
            # HARD safety floor at ≥90 (avoid double-counting; the old ≥80→38 cap
            # was removed as it double-penalised what `room` already handles).
            if exf >= 90:
                score = min(score, 22)    # → EXHAUSTED (safety floor)
        label, color = self._score_label(score)
        # Fuel STRENGTH 0..100 = |fuel imbalance| × 100 (how lopsided the
        # liquidity is). |dir| ≤ 0.1 (≤10%) → no direction; higher = stronger.
        fuel_strength = int(round(abs(signed) * 100)) if signed is not None else None
        return {'score': int(round(score)), 'label': label, 'color': color,
                'dir': live_dir, 'conflict': conflict, 'exh': exf,
                # Per-coin ММ (liq-fuel) direction — shown in its own column in
                # the ❤️ queue table. LONG / SHORT / None(=збалансований).
                'fuel_dir': fuel_dir,
                'fuel_strength': fuel_strength}

    def score_dict(self, symbol: str) -> Optional[Dict]:
        """Full SCORE verdict dict for `symbol` RIGHT NOW — same shape the
        ⏱️ Active Timers rows carry ({score,label,color,dir,conflict}) — so the
        open-positions tables (real + paper) can render the IDENTICAL badge.
        Pulls held/dir/exhaustion from the live timer when present, else scores
        with held=0 and lets the score derive its own live direction.

        Prefers the background score cache (fast, no per-call liq-map/kline work)
        so TM's get_state — called on every UI poll — stays cheap. Only falls
        back to a live compute for symbols not in the cache (e.g. manual
        positions FF doesn't scan), which are few."""
        try:
            sym = (symbol or '').upper().strip()
            if not sym:
                return None
            with self._lock:
                cached = self._score_cache.get(sym)
            if cached:
                return cached
            s = self.get_settings()
            tf = s.get('engine_candle_tf', '5m')
            with self._lock:
                t = self._timers.get(sym)
                if t:
                    held = time.time() - t.get('since', time.time())
                    tdir = t.get('dir') or 'LONG'
                    exh = t.get('exhaustion')
                else:
                    held, tdir, exh = 0.0, 'LONG', None
            dur = float(s.get('duration_minutes', 5) or 5) * 60
            return self._timer_score_for(sym, tdir, held, exh, dur, tf)
        except Exception:
            return None

    def score_snapshot(self, symbol: str) -> Optional[str]:
        """Compact, human SCORE string for `symbol` RIGHT NOW, e.g.
        'СИЛЬНЕ УТРИМАННЯ 🟢▲ 79'. Used to stamp a position at open and at close."""
        sc = self.score_dict(symbol)
        if not sc:
            return None
        arrow = '🟢▲' if sc.get('dir') == 'LONG' else (
            '🔴▼' if sc.get('dir') == 'SHORT' else '')
        warn = ' ⚠' if sc.get('conflict') else ''
        return f"{self._score_label_ua(sc['label'])} {arrow} {sc['score']}{warn}".strip()

    # ------------------------------------------------------------------
    # fuel / exhaustion measurement (cached sources only)
    # ------------------------------------------------------------------
    def _fuel_dir(self, symbol: str) -> Optional[Dict]:
        """Replicate compute_bias()'s fuel-direction math off the CACHED
        liquidation-map state (no exchange calls). Returns
        {dir, mark_price, status} or None when data is unavailable."""
        try:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            mark = None
            lst = None
            if lm:
                try:
                    prof = self._db.get_setting('liqmap_decay_profile', 'tori')
                except Exception:
                    prof = 'tori'
                lst = lm.get_state(symbol, lookback_hours=24, profile=prof)
                mark = lst.get('mark_price') if lst else None

            # Fallback: if liquidation map doesn't have mark_price, get it from market_data
            if not mark:
                try:
                    from detection.market_data import get_market_data
                    md = get_market_data()
                    if md:
                        ticker = md.get_ticker(symbol)
                        mark = ticker.get('last') if ticker else None
                except Exception:
                    pass

            if not mark or mark <= 0:
                return None

            fa = fb = 0.0
            for lev in (lst.get('levels') or []) if lst else []:
                dist = abs(lev['price'] - mark) / mark * 100.0
                if dist > 15:
                    continue
                w = lev['usd'] / (1.0 + dist / 2.0)
                if lev['price'] > mark:
                    fa += w
                else:
                    fb += w
            den = fa + fb
            fuel_dir = (fa - fb) / den if den > 0 else 0.0
            if den <= 0:
                status = None
            elif fuel_dir > FUEL_LONG_THR:
                status = 'LONG'
            elif fuel_dir < FUEL_SHORT_THR:
                status = 'SHORT'
            else:
                status = None
            return {'dir': round(fuel_dir, 3), 'mark_price': mark,
                    'status': status}
        except Exception as e:
            print(f"[FuelFilter] fuel calc error {symbol}: {e}")
            return None

    def _fuel_dir_smoothed(self, symbol: str, update: bool = False) -> Optional[Dict]:
        """Anti-twitch wrapper over _fuel_dir. Returns the SAME shape but with a
        time-smoothed `dir` (EMA over `direction_smoothing_min`) and a hysteresis
        `status` (enter ±0.15 / exit <±0.05, sticky in between).

        update=True  → advance the EMA + hysteresis latch (call ONCE per scan
                       cycle, from _tick). update=False → read-only (UI/score).
        `raw_dir`/`raw_status` carry the instantaneous values for reference."""
        fd = self._fuel_dir(symbol)
        if not fd:
            return None
        raw = fd.get('dir', 0.0)
        sym = (symbol or '').upper()
        W = float(self.get_settings().get('direction_smoothing_min', 0) or 0)
        if W <= 0:
            # Smoothing OFF → raw fuel, instant (no lag, matches the source the
            # main window reads). This is the default.
            return {'dir': raw, 'mark_price': fd.get('mark_price'),
                    'status': fd.get('status'), 'raw_dir': raw,
                    'raw_status': fd.get('status')}
        if update:
            N = max(1.0, W * 60.0 / CYCLE_SECS)      # samples in the window
            alpha = 2.0 / (N + 1.0)                   # EMA smoothing factor
            prev = self._fuel_ema.get(sym)
            ema = raw if prev is None else (alpha * raw + (1.0 - alpha) * prev)
            self._fuel_ema[sym] = ema
            # Hysteresis latch on the SMOOTHED value.
            cur = self._fuel_hyst.get(sym)
            if ema > 0.15:
                cur = 'LONG'
            elif ema < -0.15:
                cur = 'SHORT'
            elif abs(ema) < 0.05:
                cur = None
            # else: keep the current latch (sticky 0.05..0.15 zone)
            self._fuel_hyst[sym] = cur
            status = cur
        else:
            ema = self._fuel_ema.get(sym, raw)
            # Read the latch; if smoothing hasn't initialised yet, fall back to
            # the raw status so a fresh coin is still usable immediately.
            status = self._fuel_hyst.get(sym) if sym in self._fuel_hyst \
                else fd.get('status')
        return {'dir': round(ema, 3), 'mark_price': fd.get('mark_price'),
                'status': status, 'raw_dir': raw, 'raw_status': fd.get('status')}

    def get_btc_session(self) -> Dict:
        """The committed ₿ BTCUSDT session that the banner shows:
        {'dir': 'LONG'|'SHORT'|None, 'paused': bool, 'since': float}.
        Used by AutoGate 'banner' mode to mirror the banner onto the
        LONG/SHORT direction buttons."""
        with self._lock:
            return {'dir': self._btc_verdict_dir,
                    'paused': bool(self._btc_paused),
                    'since': float(self._btc_verdict_since or 0.0)}

    def _enforce_btc_flip_close(self, settings: Dict):
        """Close open trades (real + test) that the ₿ banner no longer supports.
        Mode 'flip'  → close only when the banner FULLY turned to the opposite
                       direction (active, not paused).
        Mode 'pause' → close as soon as the banner stops CONFIRMING the trade's
                       side — i.e. it went to ПАУЗА/WAIT on that side OR flipped.
        """
        mode = str(settings.get('close_on_btc_mode', '') or '').lower()
        if mode == 'off':
            # EXPLICIT off (user picked «Вимкнено») — never close, and DO NOT
            # fall back to the legacy boolean. A stale close_on_btc_flip=True in
            # the DB used to fire btc_flip closes here even though the UI showed
            # «Вимкнено» — that was the bug.
            return
        if mode not in ('pause', 'flip'):
            # mode UNSET/unknown (old config without the key) → legacy fallback.
            if settings.get('close_on_btc_flip'):
                mode = 'flip'
            else:
                return
        bdir = self._btc_verdict_dir
        if bdir not in ('LONG', 'SHORT'):
            return   # no committed session → nothing to enforce
        paused = bool(self._btc_paused)
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return

        def _should_close(side):
            # banner actively confirms this side?
            confirmed = (bdir == side) and (not paused)
            if mode == 'flip':
                # only a full, active opposite direction closes it
                return (bdir != side) and (not paused)
            # 'pause' → close whenever the side is not actively confirmed
            return not confirmed

        def _close_book(book_attr, is_real):
            book = getattr(tm, book_attr, {}) or {}
            for sym in list(book.keys()):
                p = book.get(sym)
                if not p or not _should_close(p.get('side')):
                    continue
                try:
                    if is_real:
                        if hasattr(tm, 'manual_close') and callable(tm.manual_close):
                            tm.manual_close(sym, reason='btc_flip')
                    else:
                        price = (tm._get_current_price(sym)
                                 if hasattr(tm, '_get_current_price') else None) \
                                or p.get('entry_price')
                        if hasattr(tm, '_close_shadow') and callable(tm._close_shadow):
                            tm._close_shadow(sym, price, 'btc_flip')
                    print(f"[FuelFilter] ₿-{mode} → closed {'real' if is_real else 'paper'} "
                          f"{p.get('side')} {sym} (banner {bdir}{' ПАУЗА' if paused else ''})")
                except Exception as e:
                    print(f"[FuelFilter] btc-flip close {sym}: {e}")

        _close_book('_positions', True)
        _close_book('_shadow_positions', False)

    def _update_btc_verdict(self):
        """BTC ММ *session* tracker (drives the ₿ banner + START engine + queue).

        A SESSION = a committed BTC ММ direction. The live ММ comes from the
        MAIN-WINDOW indicator (compute_bias fuel, ±0.1): dir > +0.1 → LONG,
        < −0.1 → SHORT, |dir| ≤ 0.1 → WAIT (ML збалансований).

        Session rules (per user's "сеанси"):
          • WAIT / data gap → PAUSE: keep the session direction, keep its start
            time (timer keeps counting), keep the queue. `_btc_paused = True`.
          • Same direction (LONG→WAIT→LONG) → SAME session, resumes.
          • OPPOSITE direction (LONG↔SHORT) → NEW session: reset the start time,
            CLEAR the whole ❤️ queue, start counting the other way.
        So the queue is cleared ONLY on a genuine session flip — never on a
        transient WAIT. Called once per cycle."""
        try:
            from web.flask_app import compute_bias
            d = compute_bias(self._db, 'BTCUSDT', None)
            fuel = ((d or {}).get('components') or {}).get('fuel') or {}
            fdir = fuel.get('dir')
            # BTC fuel STRENGTH 0..100 (|imbalance|×100) — fills the ₿ banner bar.
            if fdir is not None:
                self._btc_fuel_strength = int(round(abs(fdir) * 100))
            if fdir is None:
                live = None                 # data gap → treat as WAIT (pause)
            elif fdir > FUEL_LONG_THR:      # > +0.1 → LONG (як головне вікно)
                live = 'LONG'
            elif fdir < FUEL_SHORT_THR:     # < -0.1 → SHORT
                live = 'SHORT'
            else:
                live = None                 # |dir| ≤ 0.1 → збалансований → WAIT
        except Exception as e:
            print(f"[FuelFilter] BTC ММ calc error: {e}")
            return
        now = time.time()
        sess = self._btc_verdict_dir        # current session direction

        # WAIT / balanced → pause the session (keep dir, timer, queue).
        if live is None:
            if not self._btc_paused and sess:
                self._btc_paused = True
                self._persist_state()
            return

        # No session yet → start one (nothing to clear).
        if sess is None:
            self._btc_verdict_dir = live
            self._btc_verdict_since = now
            self._btc_paused = False
            self._persist_state()
            return

        # Same direction → session continues (resume from pause if needed).
        if live == sess:
            if self._btc_paused:
                self._btc_paused = False
                self._persist_state()
            return

        # OPPOSITE direction → NEW session: reset the session timer/dir only.
        # The queue is NO LONGER purged by direction on a flip — the actionable
        # side is decided per-coin (fuel / smart direction), not by the session,
        # so a session flip must not wipe coins. Queue cleanup is now handled
        # SOLELY by the TTL (queue_ttl_hours) + open / manual / TM-holds.
        # (OP-4 direction-purge retired.)
        self._btc_verdict_dir = live
        self._btc_verdict_since = now
        self._btc_paused = False
        self._persist_state()

    def _bias(self, symbol: str) -> Dict:
        """Cached FF-specific bias computation. Uses compute_bias_for_ff —
        a SOFTER variant that issues LONG/SHORT more readily than the strict
        dashboard version. This reduces the time FF spends in WAIT state.
        All FF consumers (exhaustion, verdict, panel-status) go through here
        so the numbers are consistent."""
        now = time.time()
        c = self._bias_cache.get(symbol)
        if c and (now - c.get('ts', 0)) < BIAS_TTL:
            return c.get('data') or {}
        from web.flask_app import compute_bias_for_ff
        data = compute_bias_for_ff(self._db, symbol) or {}
        self._bias_cache[symbol] = {'ts': now, 'data': data}
        return data

    def _decision_verdict(self, symbol: str, price: Optional[float] = None) -> Dict:
        """Decision-Center snapshot for `symbol`: {'verdict': 'good'|'marginal'|
        'poor'|None, 'recommended': 'LONG'|'SHORT'|'NEUTRAL'|None}. Same object
        the 🧠 badge shows on the chart — sourced from TradeManager.compute_decision()
        (→ decision_center.build_decision). Cached BIAS_TTL sec (it runs the entry
        evaluator) — call only from the engine's final open gate, never per tick.

        NB: earlier this read compute_bias(...)['decision'], but compute_bias has
        NO 'decision' key → verdict was ALWAYS None, so the «Мін. якість рішення»
        gate blocked EVERY coin whenever set to СЕРЕДНІЙ/СИЛЬНИЙ. Fixed to use the
        same source as the chart badge."""
        now = time.time()
        c = self._decision_cache.get(symbol)
        if c and (now - c.get('ts', 0)) < BIAS_TTL:
            return c.get('v') or {}
        v = {}
        try:
            tm = self._get_tm() if self._get_tm else None
            if tm and hasattr(tm, 'compute_decision') and callable(tm.compute_decision):
                px = price
                if not px:
                    f = self._fuel_dir_smoothed(symbol) or {}
                    px = f.get('mark_price')
                dec = tm.compute_decision(symbol, float(px or 0.0)) or {}
                v = {'verdict': dec.get('verdict'),
                     'recommended': dec.get('recommended')}
        except Exception as e:
            print(f"[FuelFilter] decision verdict err {symbol}: {e}")
        self._decision_cache[symbol] = {'ts': now, 'v': v}
        return v

    def _is_wait_verdict(self, symbol: str) -> bool:
        """Check if the given symbol has a WAIT verdict (unclear direction).
        Returns True if verdict is WAIT, False otherwise (or on error)."""
        try:
            verdict = self._bias(symbol).get('verdict', 'WAIT')
            return verdict == 'WAIT'
        except Exception as e:
            print(f"[FuelFilter] verdict check error {symbol}: {e}")
            return False  # on error, don't block the trade

    def get_coin_indicators(self, symbol: str) -> Dict:
        """Collect key indicators for a symbol (used by FF UI second row).
        Returns: {forecast_1h, forecast_4h, fuel_status} or partial dict on error.
        Forecast: {side, pct, confidence} or None. Fuel: 'LONG'|'SHORT'|None."""
        result = {}
        # --- Forecast 1H & 4H ---
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            cached = fe.get(symbol) if fe else None
            if cached:
                f1 = cached.get('forecast_1h') or {}
                f4 = cached.get('forecast_4h') or {}
                result['forecast_1h'] = {'side': f1.get('side', 0),
                                          'pct': f1.get('pct', 0),
                                          'confidence': f1.get('confidence', 0)} if f1.get('side') else None
                result['forecast_4h'] = {'side': f4.get('side', 0),
                                          'pct': f4.get('pct', 0),
                                          'confidence': f4.get('confidence', 0)} if f4.get('side') else None
            else:
                result['forecast_1h'] = None
                result['forecast_4h'] = None
        except Exception:
            result['forecast_1h'] = None
            result['forecast_4h'] = None
        # --- Fuel direction (smoothed + hysteresis, read-only) ---
        try:
            fuel_data = self._fuel_dir_smoothed(symbol)
            result['fuel_status'] = fuel_data.get('status') if fuel_data else None
        except Exception:
            result['fuel_status'] = None
        return result

    def get_panel_status(self, symbol: str) -> Dict:
        """Full FF decision-state for ONE symbol — used by the chart panel's
        second status row. Heavier than get_coin_indicators (computes verdict +
        exhaustion), but only ever called for the single currently-open symbol.

        Returns everything the operator needs to understand WHY FF is or isn't
        acting on this coin:
          enabled         — FF master toggle
          in_scan_list    — is this coin in FF's scan whitelist
          fuel_status     — liq-fuel direction (the timer trigger): LONG/SHORT/None
          exhaustion      — move exhaustion % for fuel-status side (entry gate)
          max_exhaustion  — entry gate threshold
          exhausted       — exhaustion > max (entry would be rejected)
          verdict         — compute_bias verdict (LONG/SHORT/WAIT)
          skip_wait       — skip_wait_coins setting
          wait_blocked    — verdict==WAIT AND skip_wait (entry/position blocked)
          timer           — {progress_pct, held_sec, dir} or None
          holding         — FF currently manages an open position on this coin
        """
        symbol = (symbol or '').upper()
        settings = self.get_settings()
        # Ensure the liq-map is tracking this coin so fuel data exists even for
        # on-demand symbols (e.g. a TradingView userscript querying an arbitrary
        # chart the bot isn't otherwise scanning). First call may be neutral
        # until the next liq-map scan cycle fills the levels.
        try:
            self._register_with_liqmap([symbol])
        except Exception:
            pass
        out = {
            'enabled': bool(settings.get('enabled', False)),
            'in_scan_list': symbol in set(self.get_scan_list()),
            'skip_wait': bool(settings.get('skip_wait_coins', False)),
            'max_exhaustion': settings.get('max_exhaustion_pct', 75),
        }
        # Fuel direction (timer trigger) — smoothed + hysteresis, read-only.
        # Also expose the ММ STRENGTH (|fuel dir|×100, 0..100) + raw dir so
        # external overlays can show the exact same ММ the bot uses.
        try:
            fuel_data = self._fuel_dir_smoothed(symbol)
            out['fuel_status'] = fuel_data.get('status') if fuel_data else None
            _fd = float((fuel_data or {}).get('dir') or 0.0)
            out['mm'] = fuel_data.get('status') if fuel_data else None
            out['mm_dir'] = round(_fd, 3)
            out['mm_str'] = round(abs(_fd) * 100.0, 1)
        except Exception:
            out['fuel_status'] = None
            out['mm'] = None
            out['mm_dir'] = 0.0
            out['mm_str'] = 0.0
        # Exhaustion for the fuel-status side (entry gate input)
        exh = None
        try:
            side = out.get('fuel_status')
            if side in ('LONG', 'SHORT'):
                exh = self._exhaustion(symbol, side)
        except Exception:
            exh = None
        out['exhaustion'] = exh
        out['exhausted'] = (exh is not None and exh > out['max_exhaustion'])
        # Verdict (WAIT gate input) — uses wl=None (watchlist NOT considered)
        try:
            out['verdict'] = self._bias(symbol).get('verdict', 'WAIT')
        except Exception:
            out['verdict'] = None
        out['wait_blocked'] = (out['skip_wait'] and out.get('verdict') == 'WAIT')
        # Timer / holding state. NEW STRATEGY: a timer exists only for an OPEN
        # FF position (starts at open, runs while the position is open). Return
        # it so the panel can show '⏱ Таймер' ONLY when there is a live trade.
        with self._lock:
            holding = symbol in self._fuel_managed
            t = self._timers.get(symbol)
            timer = None
            if t:
                held = time.time() - t.get('since', time.time())
                timer = {'dir': t.get('dir'), 'held_sec': int(held)}
        out['holding'] = holding
        out['timer'] = timer
        # ── ₿ BTC banner state (for the overlay's BTC line) ──
        try:
            bdir = self._btc_verdict_dir if self._btc_verdict_dir in ('LONG', 'SHORT') else None
            bpaused = bool(self._btc_paused and bdir)
            bstatus = 'STOP'
            if bdir:
                period = float(settings.get('start_signal_minutes', 5) or 5) * 60
                held = time.time() - (self._btc_verdict_since or 0)
                bstatus = 'START' if held >= period else 'COUNTING'
            out['btc'] = {'dir': bdir, 'paused': bpaused, 'status': bstatus,
                          'strength': int(self._btc_fuel_strength or 0)}
        except Exception:
            out['btc'] = {'dir': None, 'paused': False, 'status': 'STOP', 'strength': 0}
        # ── 💰 Funding info — only when the coin is in the «💰 Funding — ММ по
        # монетах» table (self._anomalies). Adds current funding rate + time to
        # the next settlement so the overlay can show them for funding coins. ──
        try:
            with self._lock:
                a = self._anomalies.get(symbol)
            out['funding'] = bool(a)
            out['funding_rate'] = a.get('rate') if a else None
            out['funding_next_ms'] = a.get('next_funding') if a else None
        except Exception:
            out['funding'] = False
            out['funding_rate'] = None
            out['funding_next_ms'] = None
        # ── ⚡ CTR (STC) for the overlay foot line («⚡ CTR·15M 🟢 LONG-нахил X%») ──
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            # On-demand warm so a chart/overlay coin the scanner hasn't reached
            # yet still gets CTR (not «— немає даних»).
            if fe:
                # CTR TF from the SCANNER settings (smc_settings blob), so the
                # on-demand warm uses the SAME TF the badge/scanner use.
                _ctf = '1h'
                try:
                    from detection.smc_scanner import get_smc_scanner
                    _sc = get_smc_scanner()
                    if _sc:
                        _ctf = _sc.get_settings().get('ctr_timeframe', '1h')
                except Exception:
                    _ctf = '1h'
                fe.ensure_fresh(symbol, ctr_tf=_ctf)
            c = ((fe.get(symbol) or {}).get('ctr') if fe else None) or None
            if c and c.get('stc') is not None:
                stc = float(c.get('stc'))
                lean = 'SHORT' if stc > 50 else ('LONG' if stc < 50 else None)
                out['ctr'] = {'stc': round(stc, 1), 'lean': lean,
                              'lean_pct': round(abs(stc - 50.0) / 50.0 * 100.0),
                              'tf': c.get('tf'), 'last_dir': c.get('last_dir'),
                              'age_bars': c.get('last_signal_age_bars')}
            else:
                out['ctr'] = None
        except Exception:
            out['ctr'] = None
        return out

    def _exhaustion(self, symbol: str, side: str) -> Optional[float]:
        """Move exhaustion (0..100) for `side`. Reads the EXACT value the
        dashboard's "Потенціал LONG/SHORT" panel shows — i.e. compute_bias's
        move_long['exhaustion'] / move_short['exhaustion'] — so FF's numbers
        match the panel byte-for-byte (no separate re-computation that drifts).
        Returns None on insufficient data."""
        try:
            data = self._bias(symbol)
            mv = data.get('move_long') if side == 'LONG' else data.get('move_short')
            if mv and mv.get('ok'):
                return mv.get('exhaustion')
            return None
        except Exception as e:
            print(f"[FuelFilter] exhaustion calc error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # open / close (pure delegation to TradeManager)
    # ------------------------------------------------------------------
    def _stamp_entry_meta(self, symbol: str, meta: Dict) -> None:
        """Stamp calibration fields (ff_entry_score, ff_queue_wait_sec,
        ff_ctr_at_signal/open, …) onto whichever TM book holds `symbol`, right
        after the Q2 engine opens it — so the CLOSED record carries them for
        empirical calibration (does waiting help? does ENTRY score → PnL?)."""
        try:
            tm = self._get_tm() if self._get_tm else None
            if not tm:
                return
            sym = (symbol or '').upper()
            for book, persist in (('_positions', '_persist_positions'),
                                  ('_shadow_positions', '_persist_shadow_positions')):
                d = getattr(tm, book, None)
                if isinstance(d, dict) and sym in d and isinstance(d[sym], dict):
                    d[sym].update(meta)
                    fn = getattr(tm, persist, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
        except Exception:
            pass

    def _open(self, symbol: str, side: str, fuel: Dict, settings: Dict,
              opened_by: Optional[str] = None):
        """Trigger position open via TradeManager/TestMode. Fuel filter does NOT
        store position data — it only tracks which symbols it opened and delegates
        the actual position to TM. Positions appear in Trade Manager or Test Mode
        tables based on toggle states.

        opened_by: optional label stored on the position's "Opened by" field
        (e.g. the candle-confirm attempt the auto-engine opened on). When None,
        TM uses its default ('manual_ui')."""
        print(f"[FuelFilter] _open CALLED for {symbol} {side} (timer reached 100%)")

        entry_price = fuel.get('mark_price')
        if not entry_price or entry_price <= 0:
            print(f"[FuelFilter] {symbol}: no entry price — skip open")
            return False

        # CHECK EXHAUSTION BEFORE OPENING: don't enter exhausted moves
        max_exh = settings.get('max_exhaustion_pct', 75)
        exh = self._exhaustion(symbol, side)
        if exh is not None and exh > max_exh:
            print(f"[FuelFilter] {symbol}: exhaustion {exh:.1f}% > {max_exh}% — "
                  f"rejecting open (too exhausted)")
            return False

        # CHECK WAIT VERDICT: if enabled, don't open coins in WAIT state
        if settings.get('skip_wait_coins', False):
            if self._is_wait_verdict(symbol):
                print(f"[FuelFilter] {symbol}: verdict is WAIT — "
                      f"rejecting open (skip_wait_coins enabled)")
                return False

        tm = self._get_tm() if self._get_tm else None
        if not tm:
            print(f"[FuelFilter] {symbol}: no TradeManager available — skip open")
            return False

        # Check which mode is active (TM real or Test Mode paper)
        tm_settings = tm.get_settings() if hasattr(tm, 'get_settings') and callable(tm.get_settings) else {}
        tm_enabled = tm_settings.get('enabled', False)
        test_mode = tm_settings.get('test_mode', True)

        print(f"[FuelFilter] {symbol}: TM settings: enabled={tm_enabled}, test_mode={test_mode}")

        if not tm_enabled and not test_mode:
            print(f"[FuelFilter] {symbol}: neither TM nor Test Mode enabled — skip open")
            return False

        # Prefer real TM if both are on
        is_real = tm_enabled
        mode = 'real' if is_real else 'paper'
        print(f"[FuelFilter] {symbol}: opening in {mode} mode")

        # Don't double-open: if TM already holds this symbol (real or shadow),
        # just adopt it into tracking instead of trying to open again.
        if self._tm_has_position(symbol, is_real):
            print(f"[FuelFilter] {symbol}: TM already has a position — adopting into tracking")
        else:
            print(f"[FuelFilter] {symbol}: attempting to open {side} position via TM...")
            try:
                if is_real:
                    # Real position via TM — bypass LONG/SHORT gates
                    # Fuel Filter operates independently from manual trade signals
                    print(f"[FuelFilter] {symbol}: calling tm.manual_open({symbol}, {side}, bypass_gates=True)")
                    res = tm.manual_open(symbol, side, bypass_gates=True,
                                         opened_by=opened_by)
                    print(f"[FuelFilter] {symbol}: manual_open returned: {res}")
                    if not res or not res.get('ok'):
                        reason = (res or {}).get('reason', 'unknown')
                        print(f"[FuelFilter] {symbol}: real open rejected: {reason}")
                        return False
                    # manual_open may DOWNGRADE to a paper (shadow) position when
                    # max_open_positions is reached. Respect the mode it ACTUALLY
                    # used — otherwise we'd verify/track against the wrong book
                    # (real vs shadow), the verification below would fail, and the
                    # tick loop would immediately drop the marker / close it.
                    actual_mode = res.get('mode', 'real')
                    is_real = (actual_mode == 'real')
                    mode = 'real' if is_real else 'paper'
                    if not is_real:
                        print(f"[FuelFilter] {symbol}: manual_open downgraded to "
                              f"paper (max positions reached) — tracking as paper")
                else:
                    # Paper position via Test Mode (shadow) — bypass LONG/SHORT gates
                    # Fuel Filter operates independently from manual trade signals
                    if hasattr(tm, '_open_shadow') and callable(tm._open_shadow):
                        sh_tag = opened_by or 'fuel_filter'
                        print(f"[FuelFilter] {symbol}: calling tm._open_shadow({symbol}, {side}, {entry_price}, {sh_tag!r}, bypass_gates=True)")
                        tm._open_shadow(symbol, side, entry_price, sh_tag, bypass_gates=True)
                        print(f"[FuelFilter] {symbol}: _open_shadow call completed")
                    else:
                        print(f"[FuelFilter] {symbol}: Test Mode enabled but _open_shadow not available")
                        return False
            except Exception as e:
                print(f"[FuelFilter] {symbol}: open error ({mode}): {e}")
                import traceback
                traceback.print_exc()
                return False

            # VERIFY the open actually landed before tracking. _open_shadow can
            # silently return early (e.g. LONG/SHORT entries gated off), and a
            # real open can be rejected at the order layer. If nothing landed,
            # do NOT mark _fuel_managed — otherwise the timer disappears but no
            # position exists ("trades vanish, nothing happens"). Returning
            # False makes the caller reset the timer so we retry next DURATION
            # window instead of hammering the failing open every single cycle.
            has_pos = self._tm_has_position(symbol, is_real)
            print(f"[FuelFilter] {symbol}: verification check — _tm_has_position={has_pos}")
            if not has_pos:
                print(f"[FuelFilter] {symbol}: open did not land in TM "
                      f"(gated/rejected) — NOT tracking, timer will restart")
                return False

        # Track that fuel filter opened this position (for exit condition monitoring)
        with self._lock:
            self._fuel_managed[symbol] = {
                'opened_at': time.time(),
                'side': side,
                'fuel_dir': fuel.get('dir'),
                'mode': mode,
            }
            self._persist_state()
        exh_str = f"{exh:.1f}%" if exh is not None else 'N/A'
        print(f"[FuelFilter] OPEN SUCCESS: {mode} {side} {symbol} @ {entry_price} "
              f"(fuel {fuel.get('dir')}, exhaustion {exh_str})")
        return True

    def _tm_has_position(self, symbol: str, is_real: bool) -> bool:
        """True if TradeManager currently holds a position for this symbol in
        the relevant book (real → _positions, paper → _shadow_positions)."""
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return False
        try:
            if is_real:
                book = getattr(tm, '_positions', {}) or {}
            else:
                book = getattr(tm, '_shadow_positions', {}) or {}
            return symbol in book
        except Exception:
            return False

    def _tm_position_side(self, symbol: str) -> Optional[str]:
        """Side ('LONG'/'SHORT') of the TM position for `symbol` (real first,
        then paper), or None if there is no open position."""
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return None
        try:
            p = ((getattr(tm, '_positions', {}) or {}).get(symbol)
                 or (getattr(tm, '_shadow_positions', {}) or {}).get(symbol))
            return p.get('side') if p else None
        except Exception:
            return None

    def _reverse_close_opposite(self, symbol: str) -> bool:
        """Close whichever book (real/paper) holds a position for `symbol` — used
        by the Q2 reverse (close opposite, then open the new). Returns True if a
        close was issued."""
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return False
        done = False
        try:
            if symbol in (getattr(tm, '_positions', {}) or {}) and hasattr(tm, 'manual_close'):
                tm.manual_close(symbol, reason='reverse_via_queue2')
                done = True
            if symbol in (getattr(tm, '_shadow_positions', {}) or {}) and hasattr(tm, 'manual_close_shadow'):
                tm.manual_close_shadow(symbol, reason='reverse_via_queue2')
                done = True
        except Exception as e:
            print(f"[FF-Q2] reverse-close error {symbol}: {e}")
        if done:
            with self._lock:
                self._fuel_managed.pop(symbol, None)
                self._timers.pop(symbol, None)
        return done

    def _close(self, symbol: str, exit_price: float, reason: str, is_real: bool):
        """Trigger position close via TM. Removes fuel tracking.

        For SHADOW positions a valid exit_price is required (it drives the paper
        PnL). If we don't have one we abort the close and keep tracking so the
        next tick — with a fresh price — can do it cleanly. A bad price would
        otherwise record a garbage (huge) PnL. Real closes go through Bybit so
        the price arg is irrelevant there.
        """
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return

        if not is_real and (not exit_price or exit_price <= 0):
            # Try once more to get a price for the paper close
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    ticker = md.get_ticker(symbol)
                    exit_price = ticker.get('last') if ticker else None
            except Exception:
                exit_price = None
            if not exit_price or exit_price <= 0:
                print(f"[FuelFilter] {symbol}: no price for paper close — "
                      f"deferring (reason={reason})")
                return  # keep tracking; retry next tick

        try:
            if is_real:
                # Real position — TM will close via Bybit
                if hasattr(tm, 'manual_close') and callable(tm.manual_close):
                    tm.manual_close(symbol, reason=reason)
                else:
                    print(f"[FuelFilter] TM has no manual_close method")
            else:
                # Shadow position — TM will close internally (needs exit_price)
                if hasattr(tm, '_close_shadow') and callable(tm._close_shadow):
                    tm._close_shadow(symbol, exit_price, reason)
                else:
                    print(f"[FuelFilter] TM has no _close_shadow method")
        except Exception as e:
            print(f"[FuelFilter] close error {symbol}: {e}")
            import traceback
            traceback.print_exc()

        # Remove fuel tracking. NEW STRATEGY: the timer = position lifetime, so
        # it is reset (popped) on close. The coin is gone from the base too.
        with self._lock:
            self._fuel_managed.pop(symbol, None)
            self._timers.pop(symbol, None)
            if _q_allowed(1):   # OP 1: remove from queue on position close
                self._pending.pop(symbol, None)
                self._pending2.pop(symbol, None)
            self._persist_state()
        print(f"[FuelFilter] CLOSE trigger {symbol} reason={reason}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _run(self):
        self._stop.wait(10)  # let other singletons boot
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[FuelFilter] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def _register_with_liqmap(self, symbols: List[str]):
        """Register every watchlist symbol with the liquidation-map daemon so
        it actually SCANS them. This is the root fix for "not all WATCHLIST
        coins scanned": the liq map only tracks BACKGROUND_SYMBOLS (BTC/ETH)
        plus on-demand symbols requested in the last 30 min. Without this call,
        coins nobody is actively viewing in the UI have zero OI data, so
        fuel_dir is always neutral and they never trigger. We re-request every
        tick (30 s) which keeps them well inside the 30-min on-demand TTL."""
        try:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if not lm:
                return
            for s in symbols:
                try:
                    lm.request_symbol(s)
                except Exception:
                    pass
        except Exception as e:
            print(f"[FuelFilter] liqmap register error: {e}")

    def _tick(self):
        settings = self.get_settings()
        self._last_tick_ts = time.time()
        if not settings.get('enabled'):
            return
        # BTC banner direction = main-window ММ indicator (compute_bias fuel).
        self._update_btc_verdict()
        now = time.time()

        # Auto-close trades that conflict with the ₿ banner (optional).
        try:
            self._enforce_btc_flip_close(settings)
        except Exception as e:
            print(f"[FuelFilter] btc-flip close error: {e}")


        # 💰 Funding scanner membership / rates — it has its OWN table now.
        funding_syms = self._get_funding_symbols()
        with self._lock:
            self._funding_syms = set(funding_syms)
            self._funding_rates = self._get_funding_rates()
            self._funding_next = self._get_funding_next()
            self._funding_vols = self._get_funding_volumes()
            managed = list(self._fuel_managed.keys())
            pending = list(self._pending.keys())

        # OP-10: drop queued coins that have waited longer than the TTL
        # (queue_ttl_hours) without ever opening. Bounds the now two-sided queue.
        # 0 = disabled (only manual/flip removal).
        try:
            _ttl_h = float(settings.get('queue_ttl_hours', 0) or 0)
        except (TypeError, ValueError):
            _ttl_h = 0.0
        if _ttl_h > 0 and _q_allowed(10):
            _ttl = _ttl_h * 3600.0
            with self._lock:
                _expired = [s for s, i in self._pending.items()
                            if (now - i.get('added_at', now)) > _ttl]
                for s in _expired:
                    self._pending.pop(s, None)
                    self._engine_attempts.pop(s, None)
                # Queue 2 shares the same TTL.
                _expired2 = [s for s, i in self._pending2.items()
                             if (now - i.get('added_at', now)) > _ttl]
                for s in _expired2:
                    self._pending2.pop(s, None)
                    self._engine_attempts.pop(s, None)
                if _expired or _expired2:
                    self._persist_state()
            if _expired:
                print(f"[FuelFilter] TTL {_ttl_h:.0f}h: прибрано з черги-1 "
                      f"{len(_expired)}: {', '.join(_expired)}")
            if _expired2:
                print(f"[FuelFilter] TTL {_ttl_h:.0f}h: прибрано з черги-2 "
                      f"{len(_expired2)}: {', '.join(_expired2)}")

        # Prune STALE managed markers (TM no longer holds the position) EVERY
        # tick — independent of the manage_open_positions toggle. Without this,
        # when management is OFF the markers accumulate forever and the
        # "відкрито" counter drifts (e.g. 34 while 0 positions are actually open).
        _pruned = False
        _tm_ready = bool(self._get_tm() if self._get_tm else None)
        if _tm_ready:   # skip while TM is transiently unavailable (boot)
            for sym in managed:
                tr = self._fuel_managed.get(sym)
                if not tr:
                    continue
                is_real = tr.get('mode') == 'real'
                if not self._tm_has_position(sym, is_real):
                    with self._lock:
                        self._fuel_managed.pop(sym, None)
                        self._timers.pop(sym, None)
                    _pruned = True
        if _pruned:
            with self._lock:
                managed = list(self._fuel_managed.keys())
            self._persist_state()

        # NEW STRATEGY: fuel is computed ONLY for the few relevant coins —
        # open FF positions + the waiting base (_pending) + funding-scanner
        # coins + BTC. No more whole-WATCHLIST scan.
        relevant = list(dict.fromkeys(
            ['BTCUSDT'] + managed + pending + list(funding_syms)))
        self._register_with_liqmap(relevant)

        # Advance EMA + read fuel once per cycle for each relevant coin.
        fuels = {}
        for sym in relevant:
            fuels[sym] = self._fuel_dir_smoothed(sym, update=True)

        # BTC table-row snapshot (fuel-based, like the other coins).
        bfuel = fuels.get('BTCUSDT')
        bstatus = bfuel.get('status') if bfuel else None
        with self._lock:
            self._btc_state = {
                'dir': bstatus,
                'exhaustion': (self._exhaustion('BTCUSDT', bstatus)
                               if bstatus in ('LONG', 'SHORT') else None),
                'held_sec': 0,
                'managed': 'BTCUSDT' in self._fuel_managed,
            }

        # Manage open FF positions (exit rules).
        if settings.get('manage_open_positions', True):
            for sym in managed:
                try:
                    self._manage_position(sym, settings, now, fuels.get(sym))
                except Exception as e:
                    print(f"[FuelFilter] manage error {sym}: {e}")

        # 💰 Funding fuel table — its OWN scan (separate from the base).
        try:
            self._scan_funding(settings, now, fuels)
        except Exception as e:
            print(f"[FuelFilter] funding scan error: {e}")

        # Header stats (now only over the relevant set).
        with self._lock:
            self._scan_stats = {
                'targets': len(pending),
                'scanned': sum(1 for f in fuels.values() if f and f.get('mark_price')),
                'pending': len(pending),
                'managed': len(managed),
            }
            self._persist_state()

        # Background SCORE cache (pending + managed) — keeps the UI fast.
        self._refresh_score_cache(settings)

    def _manage_position(self, symbol, settings, now, fuel):
        """Run the exit rules for one open FF-managed position. Extracted from
        the old per-symbol tick loop; behaviour unchanged. The timer is popped
        by _close on exit."""
        with self._lock:
            track = self._fuel_managed.get(symbol)
        if not track:
            return
        side = track['side']
        is_real = track.get('mode') == 'real'
        opposite = 'SHORT' if side == 'LONG' else 'LONG'
        status = fuel.get('status') if fuel else None
        mark = fuel.get('mark_price') if fuel else None
        if not mark:
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    tk = md.get_ticker(symbol)
                    mark = tk.get('last') if tk else None
            except Exception:
                pass

        # Position vanished in TM (manual close / SL / TP) → drop marker + timer.
        if not self._tm_has_position(symbol, is_real):
            with self._lock:
                self._fuel_managed.pop(symbol, None)
                self._timers.pop(symbol, None)
            return

        exh = self._exhaustion(symbol, side)
        with self._lock:
            if exh is not None:
                track['exhaustion'] = exh
                t = self._timers.get(symbol)
                if t is not None:
                    t['exhaustion'] = exh

        opened_at = track.get('opened_at', 0)
        if opened_at and (now - opened_at) < MIN_HOLD_AFTER_OPEN_SEC:
            return

        # exit: clear flip to the opposite side.
        if status == opposite:
            self._close(symbol, mark or 0.0, reason='fuel_flipped', is_real=is_real)
            return
        # exit: WAIT verdict (optional).
        if settings.get('skip_wait_coins', False) and self._is_wait_verdict(symbol):
            self._close(symbol, mark or 0.0, reason='wait_verdict', is_real=is_real)
            return
        # exit: fuel fade (sustained), with grace.
        if status != side:
            faded = track.get('faded_since')
            if not faded:
                with self._lock:
                    track['faded_since'] = now
            elif (now - faded) >= FUEL_FADE_GRACE_SEC:
                self._close(symbol, mark or 0.0, reason='fuel_faded', is_real=is_real)
                return
        else:
            if track.get('faded_since'):
                with self._lock:
                    track.pop('faded_since', None)
        # exit: ММ (fuel) strength fell below the configured minimum (optional).
        # Closes even while direction is unchanged — a fading-but-not-flipped
        # position (e.g. ММ 80% → 15%) is no longer worth holding.
        try:
            _min_close_mm = int(settings.get('manage_close_min_mm', 0) or 0)
        except (TypeError, ValueError):
            _min_close_mm = 0
        if _min_close_mm > 0 and fuel:
            _cur_mm = abs(float(fuel.get('dir') or 0.0)) * 100.0
            if _cur_mm < _min_close_mm:
                self._close(symbol, mark or 0.0,
                            reason='mm_below_min', is_real=is_real)
                return

        # exit: exhaustion reached (optional).
        if settings.get('use_potential_exit') and exh is not None \
                and exh >= settings.get('potential_threshold_pct', 95):
            self._close(symbol, mark or 0.0, reason='potential_reached', is_real=is_real)
            return
        with self._lock:
            self._persist_state()

    def _scan_funding(self, settings, now, fuels):
        """💰 Funding fuel table (own scan): for each 💰 Funding Rate Scanner
        coin, show ONLY while it currently holds fuel (no fuel → row removed).
        Stores the live funding rate, next-funding time, previous rate (for the
        rising/falling trend) and the entry threshold so the UI can draw a
        progress bar from 'Entry ≤' → −4%. Telegram entry/exit alerts on
        transitions. Stored in self._anomalies (existing table/endpoints)."""
        funding = set(self._funding_syms)
        thr = self._get_funding_threshold()
        notifier = None
        if settings.get('ff_tg_on_entry') or settings.get('ff_tg_on_exit'):
            tm = self._get_tm() if self._get_tm else None
            notifier = getattr(tm, 'notifier', None) if tm else None
        btc_line = self._btc_status_text(settings, now)
        with self._lock:
            # ММ (fuel) strength filter: a coin enters the 💰 ММ table only if
            # its strength (|fuel dir|×100) meets the selected minimum.
            try:
                fmin_mm = int(settings.get('funding_min_mm_strength', 0) or 0)
            except (TypeError, ValueError):
                fmin_mm = 0
            # Anti-spam knobs: hysteresis (keep-in-table threshold sits below the
            # enter threshold) + per-coin re-announce cooldown.
            try:
                _hyst = max(0, int(settings.get('funding_mm_hysteresis', 0) or 0))
            except (TypeError, ValueError):
                _hyst = 0
            try:
                _cool_sec = max(0, int(settings.get('funding_notify_cooldown_min', 0) or 0)) * 60
            except (TypeError, ValueError):
                _cool_sec = 0
            _keep_mm = max(0, fmin_mm - _hyst)   # lower bar to STAY in table
            _sess_mode = bool(settings.get('funding_session_mode', False))
            # Evaluate BOTH coins currently in the funding scanner AND coins
            # already in the 💰 ММ table. A coin ENTERS only from the scanner,
            # but once in, it STAYS as long as it has fuel and meets the ММ
            # filter — even if it dropped out of the 💰 Funding Rate Scanner
            # (funding normalised). It leaves ONLY when fuel is gone / ММ below
            # the keep-threshold. This fixes "паливо зникло" firing at ММ 100%.
            _in_table = {s for s, a in self._anomalies.items() if a.get('funding')}
            for sym in (funding | _in_table):
                # Muted → ignore this coin entirely (no row, no TG, no archive).
                if self._is_funding_muted(sym, now):
                    self._anomalies.pop(sym, None)
                    continue
                in_scanner = sym in funding
                fuel = fuels.get(sym) or self._fuel_dir_smoothed(sym)
                status = fuel.get('status') if fuel else None
                mark = fuel.get('mark_price') if fuel else None
                # Funding rate/next: fresh from the scanner if still there, else
                # keep the last-known values stored on the anomaly record.
                a = self._anomalies.get(sym)
                rate = self._funding_rates.get(sym)
                if rate is None and a is not None:
                    rate = a.get('rate')
                nf = self._funding_next.get(sym)
                if nf is None and a is not None:
                    nf = a.get('next_funding')
                vol = self._funding_vols.get(sym)
                # Directional AND strong enough → in table; else treated as no-ММ.
                _strength = abs(float(fuel.get('dir') or 0.0)) * 100.0 if fuel else 0.0
                _mm = int(round(_strength))

                # ── SESSION MODE (optional, like the ₿ banner) ──────────────
                if _sess_mode:
                    _dirl = status in ('LONG', 'SHORT')
                    if a is None:
                        # start a NEW session only from the scanner, directional,
                        # and above the strength filter.
                        if in_scanner and _dirl and _strength >= fmin_mm:
                            self._anomalies[sym] = {
                                'symbol': sym, 'dir': status, 'started_at': now,
                                'start_price': mark, 'holding': True, 'sess_paused': False,
                                'last_price': mark, 'last_held_sec': 0,
                                'funding': True, 'rate': rate, 'prev_rate': rate,
                                'next_funding': nf, 'entry_threshold': thr,
                                'vol24h': vol, 'mm_str': _mm,
                            }
                            self._notify_funding(notifier, sym, status, mark, btc_line,
                                                 settings, now, entered=True, strength=_mm)
                            self._funding_notify_at[sym] = now
                        continue
                    if _dirl and status != a.get('dir'):
                        # opposite flip → NEW session (the ONLY re-announce case)
                        a.update({'dir': status, 'started_at': now, 'holding': True,
                                  'sess_paused': False, 'rate': rate, 'next_funding': nf,
                                  'mm_str': _mm, 'funding': True})
                        if mark:
                            a['last_price'] = mark
                        if vol is not None:
                            a['vol24h'] = vol
                        self._notify_funding(notifier, sym, status, mark, btc_line,
                                             settings, now, entered=True, strength=_mm)
                        self._funding_notify_at[sym] = now
                    elif _dirl:
                        # same direction → hold / resume, NO announce
                        a.update({'holding': True, 'sess_paused': False, 'dir': status,
                                  'last_held_sec': int(now - a.get('started_at', now)),
                                  'prev_rate': a.get('rate'), 'rate': rate,
                                  'next_funding': nf, 'mm_str': _mm, 'funding': True})
                        if mark:
                            a['last_price'] = mark
                        if vol is not None:
                            a['vol24h'] = vol
                    elif in_scanner:
                        # WAIT while still in the funding scanner → PAUSE (keep,
                        # no announce) — the session holds its direction.
                        a.update({'holding': True, 'sess_paused': True,
                                  'rate': rate, 'next_funding': nf,
                                  'mm_str': _mm, 'funding': True})
                        if mark:
                            a['last_price'] = mark
                        if vol is not None:
                            a['vol24h'] = vol
                    else:
                        # WAIT and gone from the scanner → session finished.
                        if a.get('holding'):
                            self._notify_funding(notifier, sym, a.get('dir'),
                                                 mark or a.get('last_price'), btc_line,
                                                 settings, now, entered=False,
                                                 strength=_mm, reason='сесію завершено')
                        self._anomalies.pop(sym, None)
                        self._funding_notify_at.pop(sym, None)
                    continue
                # ── end SESSION MODE ────────────────────────────────────────

                _was_holding = bool(a and a.get('holding'))
                # Enter needs full threshold; STAY only needs the (lower) keep
                # threshold — that hysteresis stops boundary flicker.
                _thr_now = _keep_mm if _was_holding else fmin_mm
                _qualifies = status in ('LONG', 'SHORT') and _strength >= _thr_now
                # A fresh coin can only appear if it is in the scanner right now;
                # a holding coin stays on fuel alone.
                if _qualifies and (in_scanner or _was_holding):
                    if not _was_holding:
                        self._anomalies[sym] = {
                            'symbol': sym, 'dir': status, 'started_at': now,
                            'start_price': mark, 'holding': True,
                            'last_price': mark, 'last_held_sec': 0,
                            'funding': True, 'rate': rate, 'prev_rate': rate,
                            'next_funding': nf, 'entry_threshold': thr,
                            'vol24h': vol, 'mm_str': _mm,
                        }
                        # Cooldown: suppress the TG "appear" if this coin was
                        # announced < cooldown ago (row still shows in the table).
                        _last = self._funding_notify_at.get(sym, 0)
                        if _cool_sec <= 0 or (now - _last) >= _cool_sec:
                            self._notify_funding(notifier, sym, status, mark, btc_line,
                                                 settings, now, entered=True, strength=_mm)
                            self._funding_notify_at[sym] = now
                    else:
                        a['holding'] = True
                        a['dir'] = status
                        if mark:
                            a['last_price'] = mark
                        a['last_held_sec'] = int(now - a.get('started_at', now))
                        a['prev_rate'] = a.get('rate')
                        a['rate'] = rate
                        a['next_funding'] = nf
                        a['entry_threshold'] = thr
                        if vol is not None:
                            a['vol24h'] = vol
                        a['funding'] = True
                        a['mm_str'] = _mm
                else:
                    # Leaves the table ONLY here: fuel gone or ММ below keep-thr.
                    if a is not None:
                        if a.get('holding'):
                            _reason = ('бабло зникло'
                                       if status not in ('LONG', 'SHORT')
                                       else f'ММ {_mm}% нижче фільтра')
                            self._notify_funding(
                                notifier, sym, a.get('dir'),
                                mark or a.get('last_price'), btc_line,
                                settings, now, entered=False, strength=_mm,
                                reason=_reason)
                        self._anomalies.pop(sym, None)
                        self._funding_notify_at.pop(sym, None)
            # Record what happened this tick into the per-coin archive.
            try:
                self._archive_funding_pass(now)
            except Exception as e:
                print(f"[FuelFilter] funding archive pass error: {e}")
            self._persist_state()

    @staticmethod
    def _fmt_funding_in(nf_ms, now) -> str:
        """Human countdown to the next funding settlement.
        nf_ms = nextFundingTime in ms; now = time.time() seconds.
        → 'Nгод MMхв' / 'Nхв' / 'скоро' / '—'."""
        try:
            if not nf_ms:
                return '—'
            secs = int(int(nf_ms) / 1000 - now)
            if secs <= 0:
                return 'скоро'
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}год {m:02d}хв" if h > 0 else f"{m}хв"
        except (TypeError, ValueError):
            return '—'

    def _notify_funding(self, notifier, sym, d, price, btc_line, settings, now,
                        entered, strength=None, reason=None):
        """Telegram alert when a 💰 funding coin APPEARS (entered=True) or
        DISAPPEARS (entered=False) from the 💰 ММ table. Gated by the user's
        ff_tg_on_entry / ff_tg_on_exit toggles and rendered from the editable
        templates. Placeholders: {symbol} {dir} {side} {price} {funding}
        {fuel} {exhaustion} {reason} {btc} — missing → «—».

        `strength` is the LIVE ММ strength (0..100) computed by the caller from
        the current fuel direction. It is passed in because the background
        score cache (self._fuel_str) usually has no entry yet at the exact
        moment a coin first appears — which is why {fuel} used to render «—».
        Falls back to the cache only when the caller could not supply it."""
        if not notifier:
            return
        if entered and not settings.get('ff_tg_on_entry'):
            return
        if (not entered) and not settings.get('ff_tg_on_exit'):
            return
        # ENTRY-only extra filters (do NOT touch exit messages): direction and
        # minimum ММ strength. E.g. «пропускати лише LONG і з ММ ≥ 90».
        if entered:
            _edir = str(settings.get('funding_tg_entry_dir', 'any') or 'any').upper()
            if _edir in ('LONG', 'SHORT') and d != _edir:
                return
            try:
                _emin = int(settings.get('funding_tg_entry_min_mm', 0) or 0)
            except (TypeError, ValueError):
                _emin = 0
            if _emin > 0:
                _sv = strength if strength is not None else self._fuel_str.get(sym)
                if _sv is None or float(_sv) < _emin:
                    return
        try:
            rate = self._funding_rates.get(sym)
            # rate from the scanner may be gone once a coin left it — fall back
            # to the last-known value stored on the anomaly record.
            if rate is None:
                _a = self._anomalies.get(sym)
                rate = _a.get('rate') if _a else None
            if strength is None:
                strength = self._fuel_str.get(sym)
            if strength is not None:
                strength = int(round(float(strength)))
            exh = self._exhaustion(sym, d) if d in ('LONG', 'SHORT') else None
            _nf = self._funding_next.get(sym)
            if _nf is None:
                _a = self._anomalies.get(sym)
                _nf = _a.get('next_funding') if _a else None
            ctx = {
                'symbol': sym,
                'dir': d or '—', 'side': d or '—',
                'price': (self._fmt_price(price) if price else '—'),
                # get_rates() already returns funding in PERCENT (see
                # funding_monitor: stored as rate*100) → do NOT scale again.
                'funding': (f"{rate:+.4f}" if rate is not None else '—'),
                # countdown to the next funding settlement (nextFundingTime ms).
                'funding_in': self._fmt_funding_in(_nf, now),
                'fuel': (str(strength) if strength is not None else '—'),
                'exhaustion': (f"{exh:.0f}" if exh is not None else '—'),
                'reason': (reason if reason is not None
                           else ('зʼявилась у ММ' if entered else 'бабло зникло')),
                'btc': (btc_line or ''),
            }
            tpl = (settings.get('ff_tg_entry_template') if entered
                   else settings.get('ff_tg_exit_template')) or ''

            class _Safe(dict):
                def __missing__(self, k):
                    return '—'
            try:
                msg = str(tpl).format_map(_Safe(ctx))
            except Exception:
                msg = str(tpl)
            notifier.send_message(msg)
        except Exception as e:
            print(f"[FuelFilter] funding TG error {sym}: {e}")


    def _refresh_score_cache(self, settings: Dict):
        """Background SCORE computation for the displayed rows — the waiting
        base (_pending + _pending2) + open FF positions (_timers) + funding-fuel
        coins — off the request path so the UI never blocks on liq-map/kline
        work. Queue 2 MUST be included too, otherwise a coin that sits ONLY in
        Queue 2 shows empty ММ / виснаженість / SCORE columns."""
        try:
            dur = float(settings.get('duration_minutes', 5) or 0) * 60
            tf = settings.get('engine_candle_tf', '5m')
            now = time.time()
            with self._lock:
                pending = list(self._pending.items())    # (sym, {dir, added_at})
                pending2 = list(self._pending2.items())   # Queue 2 waiting base
                timers = list(self._timers.items())       # open positions
                anomalies = [(s, a.get('dir')) for s, a in self._anomalies.items()]
            # ALL open TM positions (real + paper) — so the open-position tables'
            # ММ / виснаженість columns are filled for EVERY trade, regardless of
            # how it was opened (Черга-1, Черга-2 or a direct open). Read outside
            # self._lock (separate lock on TM's side).
            tm_positions = []   # (sym, side, opened_at)
            try:
                tm = self._get_tm() if self._get_tm else None
                if tm is not None and hasattr(tm, '_lock'):
                    with tm._lock:
                        for _sym, _p in list(getattr(tm, '_positions', {}).items()):
                            tm_positions.append((_sym, _p.get('side'), _p.get('opened_at')))
                        for _sym, _p in list(getattr(tm, '_shadow_positions', {}).items()):
                            tm_positions.append((_sym, _p.get('side'), _p.get('opened_at')))
            except Exception:
                pass
            # symbol -> (dir, held_sec)
            targets = {}
            for sym, info in pending:
                targets[sym] = (info.get('dir'), 0.0)   # waiting → no fuel-hold
            for sym, info in pending2:
                targets.setdefault(sym, (info.get('dir'), 0.0))
            for sym, t in timers:
                targets[sym] = (t.get('dir'), now - t.get('since', now))
            for sym, side, oa in tm_positions:
                if sym and side in ('LONG', 'SHORT'):
                    targets[sym] = (side, now - (oa or now))   # open position wins
            for sym, d in anomalies:
                targets.setdefault(sym, (d, 0.0))
            cache = {}
            for sym, (d, held) in targets.items():
                try:
                    sc = self._timer_score_for(sym, d, held, None, dur, tf)
                    if sc:
                        cache[sym] = sc
                except Exception:
                    pass
            # Fuel-strength trend: snapshot this cycle's strengths and keep the
            # previous cycle so the UI can draw a rising/falling arrow.
            new_str = {s: sc.get('fuel_strength') for s, sc in cache.items()
                       if sc.get('fuel_strength') is not None}
            with self._lock:
                self._score_cache = cache
                self._fuel_str_prev = self._fuel_str
                self._fuel_str = new_str
        except Exception as e:
            print(f"[FuelFilter] score cache error: {e}")

    @staticmethod
    def _fmt_dur(sec) -> str:
        """Compact duration: 'Hг MMхв' / 'Mхв SSс' / 'Sс'."""
        sec = max(0, int(sec))
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        if h:
            return f"{h}г {m:02d}хв"
        if m:
            return f"{m}хв {s:02d}с"
        return f"{s}с"

    @staticmethod
    def _fmt_price(p) -> str:
        """Compact price formatter for Telegram messages."""
        try:
            p = float(p)
        except (TypeError, ValueError):
            return '—'
        if p <= 0:
            return '—'
        if p >= 100:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:.4f}"
        return f"{p:.6g}"

    def _funding_alert(self, s: Dict, now: float):
        """Telegram alerts for 💰 funding coins, evaluated LIVE so they fire
        within the fast alert cadence (not the 30s scan tick):
          • ENTRY when a funding coin is in the table (timer ≥ threshold AND ММ
            still holds its direction) — with direction, price and funding %;
          • EXIT when ММ no longer holds (live) and it leaves the table.
        Each fires once; re-armed on the opposite transition. Two-line layout."""
        if not s.get('funding_tg_alerts', False):
            self._funding_alerted.clear()
            return
        fdur = float(s.get('funding_duration_minutes', 0) or 0) * 60
        tm = self._get_tm() if self._get_tm else None
        notifier = getattr(tm, 'notifier', None) if tm else None
        # Snapshot eligible funding timers under the lock; do the LIVE fuel
        # checks (and Telegram) outside it.
        with self._lock:
            candidates = []
            for sym in list(self._funding_syms):
                if sym in self._fuel_managed or sym in self._anomalies:
                    continue
                t = self._timers.get(sym)
                if not t or (now - t.get('since', now)) < fdur:
                    continue
                candidates.append((sym, t.get('dir')))
            alerted = set(self._funding_alerted)
            managed_or_anom = set(self._fuel_managed) | set(self._anomalies)
        # Live membership: in the table only if ММ still holds the direction.
        current, dir_of, price_of = set(), {}, {}
        for sym, d in candidates:
            fuel = self._fuel_dir_smoothed(sym)
            # A clear opposite/neutral reading (with data) means ММ ended →
            # not in the table. A data gap (fuel is None) keeps it (no flapping).
            if fuel is not None and fuel.get('status') != d:
                continue
            current.add(sym)
            dir_of[sym] = d
            price_of[sym] = fuel.get('mark_price') if fuel else None
        new_entries = [sym for sym in current if sym not in alerted]
        # Exit = was alerted, now out of the table for ММ reasons (not because it
        # moved to a position / anomalies table).
        left = [sym for sym in (alerted - current) if sym not in managed_or_anom]
        with self._lock:
            self._funding_alerted = set(current)
        if not notifier:
            return
        btc_line = self._btc_status_text(s, now)   # current ₿ status

        def _line2(sym, d=None):
            rate = self._funding_rates.get(sym)
            rtxt = (f"funding {rate:+.3f}% · " if rate is not None else '')
            etxt = ''
            if d in ('LONG', 'SHORT'):
                exh = self._exhaustion(sym, d)
                if exh is not None:
                    etxt = f"🔥 {exh:.1f}% · "
            nf = self._funding_next.get(sym)
            ntxt = ''
            if nf:
                rem = int(nf / 1000 - now)
                if rem > 0:
                    ntxt = f" · ⏳ {self._fmt_dur(rem)} до funding"
            return f"{rtxt}{etxt}{btc_line}{ntxt}"

        for sym in new_entries:
            d = dir_of.get(sym)
            dtxt = '🟢 LONG' if d == 'LONG' else ('🔴 SHORT' if d == 'SHORT' else '')
            ptxt = (f" · {self._fmt_price(price_of.get(sym))}" if price_of.get(sym) else '')
            try:
                notifier.send_message(f"💰 <b>{sym}</b> {dtxt}{ptxt}\n{_line2(sym, d)}")
            except Exception as e:
                print(f"[FuelFilter] funding TG send error: {e}")
        for sym in left:
            fuel = self._fuel_dir_smoothed(sym)
            price = fuel.get('mark_price') if fuel else None
            d = fuel.get('status') if fuel else None
            ptxt = (f" · {self._fmt_price(price)}" if price else '')
            try:
                notifier.send_message(f"💰 <b>{sym}</b> ⛔ вихід{ptxt}\n{_line2(sym, d)}")
            except Exception as e:
                print(f"[FuelFilter] funding exit TG send error: {e}")

    def _btc_status_text(self, s: Dict, now: float) -> str:
        """Compact current ₿ BTCUSDT status for messages as a direction:
        '₿ 🟢 LONG' / '₿ 🔴 SHORT' when START is active, else '₿ ⚪ WAIT'
        (no confirmed start: counting, or no BTC timer)."""
        period = float(s.get('start_signal_minutes', 5) or 5) * 60
        vdir = self._btc_verdict_dir
        if vdir in ('LONG', 'SHORT') and self._btc_verdict_since:
            held = now - self._btc_verdict_since
            if held >= period:
                return '₿ 🟢 LONG' if vdir == 'LONG' else '₿ 🔴 SHORT'
        return '₿ ⚪ WAIT'

    def _btc_start_alert(self, s: Dict, now: float):
        """Telegram message when the ₿ BTCUSDT banner state MEANINGFULLY
        changes (if enabled). The state token is DIRECTION-AWARE:
          START-LONG / START-SHORT — session active in that direction,
          STOP — no session direction.
        So a session FLIP (LONG↔SHORT) now fires an alert too (the old
        START/STOP-only token stayed 'START' across a flip → silent). A WAIT
        PAUSE keeps the last token (no spam, no false STOP)."""
        if not s.get('start_signal_tg_alerts', False):
            return
        period = float(s.get('start_signal_minutes', 5) or 5) * 60
        vdir = self._btc_verdict_dir
        if vdir in ('LONG', 'SHORT') and self._btc_verdict_since:
            # Remember direction so a later STOP can name the side it ran in.
            self._btc_last_dir = vdir
            if self._btc_paused:
                return               # pause: hold last token, do not alert
            held = now - self._btc_verdict_since
            if held < period:
                return               # counting up — intermediate, no alert
            token = f"START-{vdir}"
        else:
            token = "STOP"
        prev = self._btc_start_last_alert
        if prev is None:
            # First observation — record without alerting (avoid startup spam).
            self._btc_start_last_alert = token
            return
        if token == prev:
            return
        self._btc_start_last_alert = token
        tm = self._get_tm() if self._get_tm else None
        notifier = getattr(tm, 'notifier', None) if tm else None
        if not notifier:
            return
        def _dtxt(dd):
            return '🟢 LONG' if dd == 'LONG' else ('🔴 SHORT' if dd == 'SHORT' else '')
        if token.startswith('START'):
            _d = token.split('-', 1)[1]
            _icon = '🟢' if _d == 'LONG' else '🔴'
            msg = f"{_icon} <b>BTCUSDT START</b> {_dtxt(_d)}"
        else:
            # STOP — show the direction it was running in before stopping.
            msg = f"⛔ <b>BTCUSDT STOP</b> {_dtxt(self._btc_last_dir)}".rstrip()
        try:
            notifier.send_message(msg)
            print(f"[FuelFilter] BTC TG alert sent: {token}")
        except Exception as e:
            print(f"[FuelFilter] BTC TG send error: {e}")

    # ------------------------------------------------------------------
    # public API for the dashboard
    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='fuel-filter')
        self._thread.start()
        # Auto-engine loop (BTC-START basket opener) — separate cadence.
        if not (self._engine_thread and self._engine_thread.is_alive()):
            self._engine_thread = threading.Thread(target=self._run_engine,
                                                   daemon=True, name='fuel-engine')
            self._engine_thread.start()
        # Fast alert loop — near-instant Telegram on threshold crossings.
        if not (self._alert_thread and self._alert_thread.is_alive()):
            self._alert_thread = threading.Thread(target=self._run_alerts,
                                                  daemon=True, name='fuel-alerts')
            self._alert_thread.start()
        print("[FuelFilter] daemon started")

    def stop(self):
        self._stop.set()

    def _run_alerts(self):
        """Fire BTC START/STOP and funding entry/exit Telegram alerts on a fast
        cadence (~3s) so they're near-instant — instead of waiting up to a full
        30s scan tick. Reads the live timer state (since is fixed; held grows),
        so threshold crossings are detected within seconds."""
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                s = self.get_settings()
                if s.get('enabled'):
                    now = time.time()
                    self._btc_start_alert(s, now)
                    # Funding entry/exit TG now fires from _scan_funding on
                    # transitions (the old _funding_alert membership-rescan is
                    # retired together with the watchlist scan).
            except Exception as e:
                print(f"[FuelFilter] alert loop error: {e}")
            self._stop.wait(3)

    # ------------------------------------------------------------------
    # BTC-START auto-engine: when ON and BTC banner == START, open the basket
    # of coins from the chosen tables (anomalies / active timers) in the BANNER
    # direction, via FF._open (→ _fuel_managed → exhaustion-exit + control).
    # ------------------------------------------------------------------
    def _run_engine(self):
        self._stop.wait(15)
        while not self._stop.is_set():
            try:
                self._engine_tick()
            except Exception as e:
                print(f"[FF-Engine] tick error: {e}")
            try:
                self._engine_tick_q2()   # ⚡ Queue 2 (independent SCORE+CTR)
            except Exception as e:
                print(f"[FF-Q2] tick error: {e}")
            secs = 15
            try:
                secs = max(5, int(self.get_settings().get('start_engine_scan_secs', 15)))
            except Exception:
                pass
            self._stop.wait(secs)

    def _candle_klines(self, symbol: str, tf: str):
        """Klines for candle confirmation at the configured TF. Uses the
        scanner's cached klines for free when the TF matches the scanner's TF;
        otherwise fetches a few bars via market_data with a short-TTL cache."""
        tm = self._get_tm() if self._get_tm else None
        scanner = getattr(tm, 'scanner', None) if tm else None
        # Free path: scanner already holds klines at its own TF.
        try:
            if scanner and hasattr(scanner, 'get_timeframe') \
                    and scanner.get_timeframe() == tf:
                kl = scanner._get_cached_klines(symbol)
                if kl:
                    return kl
        except Exception:
            pass
        # Custom TF → fetch (cached ~20s to avoid hammering on each engine tick).
        now = time.time()
        key = (symbol, tf)
        c = self._candle_cache.get(key)
        if c and (now - c[0]) < 20:
            return c[1]
        kl = None
        try:
            from detection.market_data import get_market_data
            md = get_market_data()
            if md:
                # 30 bars → enough history for a graded ATR-normalised momentum
                # (needs ATR over ~14 bars), not just the last 2.
                kl = md.fetch_klines(symbol, limit=30, interval=tf)
        except Exception:
            kl = None
        if kl:
            self._candle_cache[key] = (now, kl)
        return kl

    def _candle_confirms(self, symbol: str, side: str, tf: str = '5m'):
        """Pre-entry candle confirmation: is recent price action moving in
        `side`? Uses the last 2 CLOSED bars at the given TF — BOTH must move in
        the direction (15m × 2 = ~30-min impulse window). Returns:
          True  — confirmed (both last 2 closed bars in the direction)
          False — against / mixed
          None  — no kline data yet (caller treats as not-confirmed → waits)."""
        kl = self._candle_klines(symbol, tf)
        if not kl or len(kl) < 3:
            return None
        last2 = kl[:-1][-2:]   # drop the still-forming bar, take last 2 closed
        if len(last2) < 2:
            return None
        ups = sum(1 for k in last2 if float(k.get('p', 0)) > float(k.get('o', 0)))
        downs = sum(1 for k in last2 if float(k.get('p', 0)) < float(k.get('o', 0)))
        if side == 'LONG':
            return ups >= 2
        if side == 'SHORT':
            return downs >= 2
        return False

    def _ctr_snapshot(self, symbol: str) -> Optional[Dict]:
        """Compact CTR state for the ❤️ queue's CTR column — read from the
        forecast-engine cache (cheap, no compute). Returns {stc, last_dir,
        last_signal_age_bars} or None when unavailable."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe:
                c = (fe.get(symbol) or {}).get('ctr') or {}
                if c.get('stc') is not None:
                    return {'stc': c.get('stc'),
                            'last_dir': c.get('last_dir'),
                            'last_signal_age_bars': c.get('last_signal_age_bars')}
        except Exception:
            pass
        return None

    def _ctr_gate_check(self, symbol: str, d: str, s: Dict) -> Optional[str]:
        """⚡ CTR OPEN gate. `d` = candidate dir 'LONG'/'SHORT'. Returns None if
        the trade may open, else a UA reason string. CTR is used as an ENTRY-
        TIMING filter for the momentum-continuation FF entry:
          • anti_extreme («не входити проти нахилу») — block a trade that goes
            AGAINST a reversal lean of ≥ ctr_gate_lean_pct %: block LONG when
            SHORT-нахил ≥ X% (overbought), block SHORT when LONG-нахил ≥ X%
            (oversold). Lean% = |STC−50|/50·100 (same as the CTR column).
          • fresh_cross  — require a fresh CTR crossover in `d` (strength-filtered
            last_dir == d AND age ≤ max).
        Fails OPEN (returns None) when CTR data is unavailable — the gate should
        never silently freeze the engine on missing forecast data."""
        mode = str(s.get('ctr_gate_mode', 'off') or 'off').lower()
        if mode == 'off':
            return None
        ctr = None
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if fe:
                ctr = (fe.get(symbol) or {}).get('ctr')
        except Exception:
            ctr = None
        if not ctr or ctr.get('stc') is None:
            return None   # no CTR data → don't block
        try:
            stc = float(ctr.get('stc'))
        except (TypeError, ValueError):
            return None
        # Reversal-lean side (50-split, same as the CTR column): stc>50 → SHORT-
        # нахил, stc<50 → LONG-нахил. No thresholds — just the side.
        lean_side = 'SHORT' if stc > 50 else ('LONG' if stc < 50 else None)
        lean = abs(stc - 50.0) / 50.0 * 100.0
        # 1) «Нахил у бік угоди» — the CTR lean must point the trade's way.
        if mode in ('anti_extreme', 'both'):
            if lean_side is not None and lean_side != d:
                return f'CTR: нахил {lean_side} {lean:.0f}% не в бік {d}'
        # 2) Fresh aligned crossover — require a recent CTR turn in our direction.
        if mode in ('fresh_cross', 'both'):
            try:
                max_age = int(s.get('ctr_gate_max_age_bars', 3) or 3)
            except (TypeError, ValueError):
                max_age = 3
            last_dir = ctr.get('last_dir')
            age = ctr.get('last_signal_age_bars')
            if last_dir != d:
                return f'CTR: немає свіжого кросоверу {d} (останній {last_dir or "—"})'
            if age is None or age > max_age:
                return f'CTR: кросовер {d} застарілий (age {age if age is not None else "—"}>{max_age})'
        return None

    def _ctr_lean_side(self, symbol: str) -> Optional[str]:
        """CTR lean side (50-split, same as the UI): 'LONG' if STC<50, 'SHORT'
        if STC>50, else None. Reads the cached forecast-engine CTR."""
        return self._ctr_lean(symbol)[0]

    def _ctr_lean(self, symbol: str):
        """CTR lean as (side, pct): side is 'LONG'/'SHORT'/None (50-split), pct is
        the lean strength |STC−50|/50·100 (0..100). (None, 0.0) when no CTR data."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return (None, 0.0)
            stc = ((fe.get(symbol) or {}).get('ctr') or {}).get('stc')
            if stc is None:
                return (None, 0.0)
            stc = float(stc)
            side = 'SHORT' if stc > 50 else ('LONG' if stc < 50 else None)
            return (side, abs(stc - 50.0) / 50.0 * 100.0)
        except Exception:
            return (None, 0.0)

    def _score_ctr_detail(self, symbol: str) -> str:
        """Compact «SCORE … | CTR …» string for the 🧾 log ДЕТАЛІ column, so a
        queued/dropped record carries the exact SCORE + CTR readings at that
        moment. Best-effort — returns '' on any failure (never breaks logging)."""
        parts = []
        try:
            sc = self.score_dict(symbol) or {}
            if sc.get('score') is not None:
                lbl = self._score_label_ua(sc.get('label'))
                sdir = sc.get('dir') or ''
                parts.append(f"SCORE {int(sc['score'])} {lbl}"
                             + (f"·{sdir}" if sdir else ''))
        except Exception:
            pass
        try:
            band = float(self.get_settings().get('queue2_ctr_neutral_pct', 10) or 0)
            state, stc, pct = self._ctr_state(symbol, band)
            if state == 'none':
                parts.append("CTR —")            # truly no data (STC is None)
            elif state == 'neutral':
                parts.append(f"CTR STC {stc:.0f} · нейтральний")
            else:
                parts.append(f"CTR STC {stc:.0f} · нахил {state} {pct:.0f}%")
        except Exception:
            pass
        return ' | '.join(parts)

    def _engine_tick_q2(self):
        """⚡ Queue 2 engine — INDEPENDENT of ₿ START and the main LONG/SHORT
        buttons. A signal reaches Q2 only after the SIGNAL-time CTR gate in
        intercept() (CTR lean == signal side). Here each queued signal (dir =
        signal side) is HELD until BOTH:
          • SCORE == STRONG HOLD in the signal direction, AND
          • the CTR lean still points the signal's way,
        then it opens (real/test per TM). Otherwise it waits. If CTR ≠ signal at
        signal time the signal was never queued (dropped in intercept). Eject
        rules (both opt-in via settings): queue2_eject_ctr drops a waiting coin
        when the CTR lean flips OPPOSITE by ≥ queue2_eject_ctr_pct %;
        queue2_eject_choch (handled in intercept) drops it on an opposite CHoCH."""
        s = self.get_settings()
        if not s.get('enabled') or not s.get('queue2_enabled'):
            return
        with self._lock:
            items = list(self._pending2.items())
        for sym, info in items:
            d = info.get('dir')
            if d not in ('LONG', 'SHORT'):
                continue
            if sym in self._fuel_managed:
                continue
            # ⏳ TTL — a queued signal that never opened expires (logged) instead
            # of lingering for hours until an opposite CHoCH ejects it.
            ttl_h = float(s.get('queue2_ttl_hours', 6) or 0)
            if ttl_h > 0:
                _added = float(info.get('added_at') or 0)
                if _added and (time.time() - _added) > ttl_h * 3600:
                    with self._lock:
                        self._pending2.pop(sym, None)
                        self._persist_state()
                    self._engine_skip.pop(sym, None)
                    print(f"[FF-Q2] протерміновано {sym} {d}: у черзі > {ttl_h:.0f}год")
                    try:
                        from detection.activity_log import log_activity
                        log_activity(sym, 'ejected',
                                     f'Черга-2: протерміновано (у черзі > {ttl_h:.0f}год без відкриття)',
                                     side=d, source='Q2')
                    except Exception:
                        pass
                    continue
            # TM already holds this coin?
            #   • SAME direction → drop (dedup, no duplicate).
            #   • OPPOSITE direction → if «reverse via Queue 2» is ON, DON'T drop:
            #     let it fall through and pass the full algorithm (SCORE+CTR); the
            #     open step below closes the opposite trade first (reverse). If the
            #     setting is OFF, keep the old behaviour (drop).
            _reverse_on = bool(s.get('queue2_reverse_via_queue', True))
            if self._tm_has_position(sym, True) or self._tm_has_position(sym, False):
                _pside = self._tm_position_side(sym)
                if _pside == d or not _reverse_on:
                    with self._lock:
                        self._pending2.pop(sym, None)
                    self._engine_skip.pop(sym, None)
                    continue
                # opposite position + reverse enabled → keep waiting for the
                # algorithm; the reverse-close happens at the open step.
            # ⚡ CTR-flip removal (opt-in): if enabled, and the CTR lean turned to
            # the OPPOSITE side by ≥ queue2_eject_ctr_pct %, drop the coin from
            # Queue 2 (checked FIRST — even before SCORE, so a flip is caught
            # while SCORE is still weak). OFF by default → an opposite/neutral
            # CTR just makes the coin wait. Neutral (None) never ejects.
            lean, lean_pct = self._ctr_lean(sym)
            opp = 'SHORT' if d == 'LONG' else 'LONG'
            if s.get('queue2_eject_ctr') and lean == opp \
                    and lean_pct >= float(s.get('queue2_eject_ctr_pct', 20) or 0):
                with self._lock:
                    self._pending2.pop(sym, None)
                    self._persist_state()
                self._engine_skip.pop(sym, None)
                print(f"[FF-Q2] видалено {sym}: CTR розвернувся на {lean} {lean_pct:.0f}% (проти {d})")
                try:
                    from detection.activity_log import log_activity
                    log_activity(sym, 'ejected', f'Черга-2: CTR розвернувся на {lean} {lean_pct:.0f}% (проти {d})', side=d, source='Q2')
                except Exception:
                    pass
                continue
            # ⭐ OPEN gate (professional, deterministic): live ENTRY score ≥ the
            # open threshold AND CTR no longer OPPOSING (aligned, or neutral in
            # the dead-zone = the pullback is done). Opposite CTR keeps waiting.
            band = float(s.get('queue2_ctr_neutral_pct', 10) or 0)
            state, stc_v, pct_v = self._ctr_state(sym, band)
            entry = self._entry_score_for(sym, d, info.get('kind'))
            open_min = int(s.get('queue2_open_min_entry_score', 60) or 0)
            if entry['score'] < open_min:
                self._engine_skip[sym] = (f'Черга-2: ENTRY {entry["score"]}·{d} < {open_min} '
                                          f'— чекаємо кращий сетап')
                continue
            if state == opp:
                self._engine_skip[sym] = (f'Черга-2: CTR ще проти ({state} {pct_v:.0f}%) '
                                          f'— чекаємо розвороту CTR')
                continue
            # Optional main-buttons gate (default OFF): when queue2_use_buttons is
            # ON, Queue 2 opens only for a direction the LONG/SHORT buttons allow.
            if s.get('queue2_use_buttons'):
                allow_long, allow_short = self._entry_gates()
                if (d == 'LONG' and not allow_long) or (d == 'SHORT' and not allow_short):
                    self._engine_skip[sym] = f'Черга-2: кнопка {d} вимкнена — чекаємо'
                    continue
            # Optional ₿ BTCUSDT-banner gate (default OFF): open only in the
            # committed ₿ session direction; ₿ WAIT/ПАУЗА or no dir → hold.
            # NOTE(calibration — DEFERRED, decide with data): today an opposite-
            # side signal is still QUEUED and only BLOCKED here at open (it waits
            # until TTL/eject). OPEN QUESTION for the analysis phase: should we
            # instead DROP the opposite side at SIGNAL time (in intercept) when a
            # directional gate (₿ / buttons) is active — no point filling the
            # queue with a side that can't open? Buttons = manual intent (likely
            # yes); ₿ can flip soon (maybe hold-until-flip). Resolve by looking at
            # how the ₿ bar behaved during trades (chronology btc_dir/btc_paused).
            if s.get('queue2_use_btc'):
                _sess = self.get_btc_session() or {}
                _bdir, _bpaused = _sess.get('dir'), bool(_sess.get('paused'))
                if not _bdir:
                    self._engine_skip[sym] = 'Черга-2: ₿ сеанс без напрямку — чекаємо'
                    continue
                if _bpaused:
                    self._engine_skip[sym] = f'Черга-2: ₿ {_bdir} · ПАУЗА — чекаємо'
                    continue
                if d != _bdir:
                    self._engine_skip[sym] = f'Черга-2: ₿ сеанс {_bdir} ≠ {d} — чекаємо'
                    continue
            # Both align → open (₿ START independent; buttons optional per setting).
            fuel = self._fuel_dir_smoothed(sym)
            if not fuel or not fuel.get('mark_price'):
                self._engine_skip[sym] = 'Черга-2: немає ціни'
                continue
            # ⭐ Decision-Center floor (data-driven, runs LAST — only for coins that
            # already passed every cheaper gate, so the heavier compute_decision is
            # rare). CALIBRATED 2026-07-13: dec_score band [40,60)=+2.79% vs
            # [0,20)=-0.55% / [20,40)=-0.37% → require dec ≥ floor. Fails OPEN when
            # Decision data is unavailable (never freeze on missing data).
            _dec_floor = float(s.get('queue2_open_min_dec_score', 20) or 0)
            if _dec_floor > 0:
                _decc = self._decision_compact(sym, fuel.get('mark_price'), d)
                _dscore = (_decc or {}).get('score')
                if _dscore is not None and float(_dscore) < _dec_floor:
                    self._engine_skip[sym] = (f'Черга-2: Decision {int(_dscore)}·{d} '
                                              f'< {int(_dec_floor)} — чекаємо')
                    continue
            # 🔄 REVERSE: an OPPOSITE position exists and the signal has FULLY
            # passed Queue 2 → close the opposite trade FIRST, then open the new.
            _did_reverse = False
            _pside = self._tm_position_side(sym)
            if _reverse_on and _pside and _pside != d:
                _did_reverse = self._reverse_close_opposite(sym)
                if _did_reverse:
                    print(f"[FF-Q2] 🔄 реверс {sym}: закрито {_pside} → відкриваємо {d}")
                    try:
                        from detection.activity_log import log_activity
                        log_activity(sym, 'closed', f'Черга-2 РЕВЕРС: закрито {_pside} (сигнал {d} пройшов Чергу-2)', side=_pside, source='Q2')
                    except Exception:
                        pass
            _ob = '⚡ Q2 РЕВЕРС' if _did_reverse else '⚡ Q2 ENTRY+CTR'
            try:
                opened = self._open(sym, d, fuel, s, opened_by=_ob)
            except Exception as e:
                print(f"[FF-Q2] open error {sym}: {e}")
                continue
            if opened:
                with self._lock:
                    self._timers[sym] = {'dir': d, 'since': time.time(),
                                         'start_price': fuel.get('mark_price')}
                    self._pending2.pop(sym, None)
                self._engine_skip.pop(sym, None)
                # Queue wait (signal→open latency) — a key calibration metric.
                _wait = time.time() - float(info.get('added_at') or time.time())
                _wlbl = (f"{_wait/3600:.1f}год" if _wait >= 3600
                         else f"{int(_wait/60)}хв")
                _ctr_lbl = (state if state in ('LONG', 'SHORT') else 'нейтральний')
                # MATURE Decision Center at open — recorded for the calibration
                # comparison (FF ENTRY score vs Decision Center → realised PnL).
                _dec = self._decision_compact(sym, fuel.get('mark_price'), d)
                _mtf = self._ctr_confluence(sym, d)
                # Stamp calibration fields onto the trade record (survives to close).
                self._stamp_entry_meta(sym, {
                    'ff_entry_score': entry['score'],
                    'ff_queue_wait_sec': int(_wait),
                    'ff_ctr_at_signal': info.get('ctr_signal'),
                    'ff_ctr_stc_signal': info.get('ctr_stc_signal'),
                    'ff_ctr_at_open': state,
                    'ff_ctr_stc_open': stc_v,
                    'ff_kind': info.get('kind'),
                    'ff_dec_score': (_dec or {}).get('score'),
                    'ff_dec_reco': (_dec or {}).get('reco'),
                    'ff_dec_verdict': (_dec or {}).get('verdict'),
                    'ff_ctr_mtf_align': (_mtf or {}).get('align'),
                    'ff_ctr_mtf_trend': (_mtf or {}).get('trend'),
                    'ff_ctr_mtf_timing': (_mtf or {}).get('timing'),
                })
                print(f"[FF-Q2] opened {d} {sym} (ENTRY {entry['score']}, CTR {state}, чекав {_wlbl})")
                try:
                    from detection.activity_log import log_activity
                    log_activity(sym, 'opened',
                                 f'Черга-2 відкрито: ENTRY {entry["score"]}·{d} + CTR {_ctr_lbl} · чекав {_wlbl}',
                                 side=d, source='Q2',
                                 extra={'entry_score': entry['score'],
                                        'ctr_state': state, 'ctr_stc': stc_v,
                                        'queue_wait_sec': int(_wait),
                                        'ctr_at_signal': info.get('ctr_signal'),
                                        'kind': info.get('kind'),
                                        'price': fuel.get('mark_price'),
                                        'dec': _dec, 'ctr_mtf': _mtf})
                except Exception:
                    pass

    def _engine_tick(self):
        s = self.get_settings()
        # Snapshot the attempt counters so we only hit the DB when they actually
        # change this tick (they're persisted so the "🕯️ Спроби" column and the
        # "Opened by" attempt number survive a bot restart).
        _att_before = dict(self._engine_attempts)

        def _persist_attempts_if_changed():
            if self._engine_attempts != _att_before:
                self._persist_state()

        # NEW STRATEGY: candidates come from the intercepted base (_pending),
        # NOT from a watchlist scan. The engine is NEVER idle while FF is ON —
        # the ₿ START toggle only picks HOW it triggers:
        #   • ₿ START ON  (start_engine_enabled) — BTC-SESSION mode: open only
        #     while the ₿ BTCUSDT session holds a direction ≥ threshold (START)
        #     and is not paused.
        #   • ₿ START OFF                         — MAIN-BUTTONS mode: open
        #     immediately per the main LONG/SHORT buttons (the "головний банер"),
        #     no BTC trigger — the bot keeps working instead of standing idle.
        # In BOTH: eligible coins are decided by the MAIN LONG/SHORT buttons and
        # a coin opens only when its live fuel matches its OWN signal direction.
        btc_mode = bool(s.get('start_engine_enabled'))
        # engine_mode: 'off' (FF disabled) | 'btc' (₿ START) | 'buttons' (banner)
        self._engine_mode = 'off' if not s.get('enabled') else ('btc' if btc_mode else 'buttons')
        if not s.get('enabled'):
            self._engine_attempts.clear()   # FF off → engine idle, reset counters
            self._engine_gate = 'двигун вимкнено (Fuel Auto-Filter off)'
            self._engine_skip = {}
            _persist_attempts_if_changed()
            return
        now = time.time()

        if btc_mode:
            # ₿ START gate: the SESSION must hold a direction ≥ threshold.
            if self._btc_verdict_dir not in ('LONG', 'SHORT') or not self._btc_verdict_since:
                self._engine_attempts.clear()
                self._engine_gate = 'BTC-сеанс без напрямку (STOP)'
                self._engine_skip = {}
                _persist_attempts_if_changed()
                return
            # Do NOT open while the session is PAUSED (live ML is WAIT/balanced).
            # The session, its timer and the queue stay alive — opening simply
            # waits until the ML resumes the session direction.
            if self._btc_paused:
                self._engine_gate = 'BTC на ПАУЗІ (WAIT) — двигун чекає, поки ML поверне напрямок сеансу'
                self._engine_skip = {}
                _persist_attempts_if_changed()
                return
            period = float(s.get('start_signal_minutes', 5) or 5) * 60
            if (now - self._btc_verdict_since) < period:
                self._engine_attempts.clear()
                self._engine_gate = (f'BTC ще не START — чекає ще '
                                     f'{int(period - (now - self._btc_verdict_since))}s до сигналу')
                self._engine_skip = {}
                _persist_attempts_if_changed()
                return

        # Candidates = waiting base from BOTH queues (each gated by its own
        # enable toggle), filtered by the MAIN LONG/SHORT buttons. Each candidate
        # is tagged with its source queue (1 or 2) so opens pop from the right
        # one. Queue 1 takes precedence on a symbol present in both.
        # This engine handles ONLY Queue 1 (buttons/₿ START gated). Queue 2 has
        # its own independent engine (_engine_tick_q2, SCORE+CTR filter).
        allow_long, allow_short = self._entry_gates()
        q1_on = bool(s.get('queue1_enabled', True))
        if not q1_on:
            self._engine_gate = 'Черга-1 вимкнена'
            self._engine_attempts.clear()
            _persist_attempts_if_changed()
            return
        # 🧭 Smart direction: derive the OPEN side from live fuel (not the queued
        # signal side). When on, we DON'T pre-filter candidates by signal dir /
        # buttons here — the direction is decided per-coin inside the loop from
        # fuel, then all gates (incl. buttons) apply to THAT side.
        smart = bool(s.get('engine_smart_direction', False))
        cand = {}   # sym -> (waited_sec, signal_dir, queue_num)
        with self._lock:
            for sym, info in self._pending.items():
                d = info.get('dir')
                if d not in ('LONG', 'SHORT'):
                    continue
                if not smart:
                    if d == 'LONG' and not allow_long:
                        continue
                    if d == 'SHORT' and not allow_short:
                        continue
                cand[sym] = (now - info.get('added_at', now), d, 1)

        mode_lbl = "BTC-START" if btc_mode else "ЗА КНОПКАМИ"
        # Mode prefix for the gate banner so the operator sees HOW we work now.
        _mode_pfx = ('₿ START (за банером BTC)' if btc_mode
                     else 'за кнопками головного банера')
        if not cand:
            self._engine_attempts.clear()
            self._engine_gate = (f'{_mode_pfx}: 0 кандидатів — головні кнопки '
                                 f'ЛОНГ={allow_long} ШОРТ={allow_short}: монети в черзі '
                                 f'не проходять за напрямком (увімкни потрібну кнопку)')
            self._engine_skip = {}
            _persist_attempts_if_changed()
            print(f"[FF-Engine] {mode_lbl} · 0 кандидатів (кнопки L={allow_long} S={allow_short})")
            return

        dur = float(s.get('duration_minutes', 5)) * 60
        max_exh = float(s.get('max_exhaustion_pct', 75) or 75)
        tf = s.get('engine_candle_tf', '5m')
        trace = []

        # Open oldest-waiting first. `qnum` = source queue (1 or 2).
        def _pop_from_queue(_sym, _q):
            (self._pending if _q == 1 else self._pending2).pop(_sym, None)
        for sym, (held, d, qnum) in sorted(cand.items(), key=lambda kv: -kv[1][0]):
            if sym in self._fuel_managed:
                trace.append(f"{sym}:managed")
                continue   # already managed (no duplicate)
            # Skip if TM already holds this coin (real OR paper) — don't dup.
            if self._tm_has_position(sym, True) or self._tm_has_position(sym, False):
                trace.append(f"{sym}:вже-в-угодах")
                if _q_allowed(3):   # OP 3: TM already holds the coin
                    with self._lock:
                        _pop_from_queue(sym, qnum)
                continue
            fuel = self._fuel_dir_smoothed(sym)
            if not fuel or not fuel.get('mark_price'):
                trace.append(f"{sym}:немає-ціни")
                continue
            # 🧭 Smart direction: OPEN side = live fuel direction (not signal).
            if smart:
                fd = fuel.get('status')
                if fd not in ('LONG', 'SHORT'):
                    trace.append(f"{sym}:паливо нейтральне — напрямок не визначено")
                    continue
                d = fd   # override the queued signal side
                # Button/banner gate for the DERIVED direction.
                if (d == 'LONG' and not allow_long) or (d == 'SHORT' and not allow_short):
                    trace.append(f"{sym}:кнопки: {d} вимкнено")
                    continue
            # GATE: fuel must match the (signal OR derived) direction. In smart
            # mode d == fuel already, so this passes; in classic mode it enforces
            # the queued signal side.
            if fuel.get('status') != d:
                trace.append(f"{sym}:паливо {fuel.get('status') or 'нейтр'} ≠ сигнал {d}")
                continue
            # GATE: minimum ММ (fuel) STRENGTH — |fuel dir|×100 ≥ setting.
            # Separate threshold per direction (LONG / SHORT); legacy single
            # key is the fallback when a per-direction one isn't set.
            _min_mm = 0
            try:
                _legacy = int(s.get('engine_min_mm_strength', 0) or 0)
                _dir_key = ('engine_min_mm_strength_long' if d == 'LONG'
                            else 'engine_min_mm_strength_short')
                _min_mm = int(s.get(_dir_key, _legacy) or 0)
            except (TypeError, ValueError):
                _min_mm = 0
            if _min_mm > 0:
                _strength = abs(float(fuel.get('dir') or 0.0)) * 100.0
                if _strength < _min_mm:
                    trace.append(f"{sym}:ММ{_strength:.0f}%<{_min_mm}%")
                    continue
            # Exhaustion gate (same as _open) — surfaced HERE so it's visible and
            # does NOT silently waste a candle check. Too-exhausted coins are
            # skipped without counting an attempt (the move is just too far gone).
            exh = self._exhaustion(sym, d)
            if exh is not None and exh > max_exh:
                trace.append(f"{sym}:виснаж{exh:.0f}%>{max_exh:.0f}%")
                continue
            # ⚡ CTR entry-timing gate (cheap — reads the cached forecast). Blocks
            # bad-timed entries (into the opposite cycle extreme) and, in
            # fresh_cross mode, requires a recent CTR turn in our direction.
            _ctr_reason = self._ctr_gate_check(sym, d, s)
            if _ctr_reason:
                trace.append(f"{sym}:{_ctr_reason}")
                continue
            # (Candle-confirmation gate retired — opens trigger on the fuel↔
            # direction match itself.)
            # SCORE gate: only open when the coin's SCORE is STRONG HOLD AND its
            # SCORE direction matches the candidate direction.
            if s.get('engine_require_strong_hold', False):
                # Use the SAME SCORE shown in the ❤️ queue (background cache) so
                # the reason can NEVER contradict the badge the operator sees.
                # (A fresh recompute here used a different held/exhaustion moment
                # and could say WEAK while the row's badge showed STRONG HOLD.)
                # Fall back to a fresh compute only if the cache has no entry yet.
                sc = self._score_cache.get(sym) or self._timer_score_for(sym, d, held, exh, dur, tf)
                if sc.get('label') != 'STRONG HOLD' or sc.get('dir') != d:
                    trace.append(f"{sym}:SCORE {self._score_label_ua(sc.get('label'))}·{sc.get('dir')} ≠ треба СИЛЬНЕ УТРИМАННЯ·{d}")
                    continue
            # Decision-Center quality gate (LAST — heaviest). Only evaluated for
            # a candidate that already passed every cheap gate and is about to
            # open, so the expensive compute_bias runs at most a few times per
            # START event (never per-coin-per-tick). 'any' skips it entirely.
            _min_dec = str(s.get('engine_min_decision', 'any') or 'any').lower()
            if _min_dec in ('marginal', 'good'):
                _dc = self._decision_verdict(sym, fuel.get('mark_price'))
                _vd = _dc.get('verdict')
                _rec = _dc.get('recommended')
                _rank = {'poor': 0, 'marginal': 1, 'good': 2}
                _need = 2 if _min_dec == 'good' else 1
                # BOTH must hold: quality tier ≥ selected AND Decision Center
                # recommends the SAME direction as the candidate.
                if _rank.get(_vd, 0) < _need or _rec != d:
                    _UA = {'poor': 'СЛАБКИЙ', 'marginal': 'СЕРЕДНІЙ', 'good': 'СИЛЬНИЙ'}
                    trace.append(f"{sym}:рішення {_UA.get(_vd, _vd or '—')}·{_rec or '—'} "
                                 f"≠ треба {_UA.get(_min_dec, _min_dec)}·{d}")
                    continue
            try:
                # _open routes through TM (bypass gates, like Alerts but ignoring
                # the LONG/SHORT master + SMC filters) AND registers the position
                # in _fuel_managed so exhaustion-exit + control manage it.
                # The "Opened by" field records which candle-confirm attempt the
                # engine opened on (failed checks bump _engine_attempts; opening
                # on the first check is attempt #1) AND the coin's ММ timer value
                # at the moment of opening.
                # "Opened by" records the EXHAUSTION at the moment of entry.
                opened = self._open(
                    sym, d, fuel, s,
                    opened_by=(f"🔥 Exhaust {exh:.1f}%" if exh is not None else "🔥 FF"))
                if opened:
                    # Timer starts NOW (at open) and runs while the position is
                    # open; _close resets it. Coin leaves the waiting base.
                    with self._lock:
                        self._timers[sym] = {'dir': d, 'since': now,
                                             'start_price': fuel.get('mark_price')}
                        if _q_allowed(2):   # OP 2: engine opened the trade
                            _pop_from_queue(sym, qnum)
                    self._engine_attempts.pop(sym, None)   # opened → reset
                    trace.append(f"{sym}:✅ВІДКРИТО {d}")
                    print(f"[FF-Engine] opened {d} {sym} ({mode_lbl}, exh="
                          f"{('%.1f%%' % exh) if exh is not None else '—'})")
                    try:
                        from detection.activity_log import log_activity
                        log_activity(sym, 'opened', f'Черга-1 · {mode_lbl}'
                                     + (f' · виснаж {exh:.0f}%' if exh is not None else ''),
                                     side=d, source='Q1')
                    except Exception:
                        pass
                else:
                    trace.append(f"{sym}:TM відхилив відкриття (Trade Manager / Test Mode вимкнені?)")
                    try:
                        from detection.activity_log import log_activity
                        log_activity(sym, 'rejected', 'Черга-1: TM відхилив відкриття (TM/Test Mode вимкнені?)', side=d, source='Q1')
                    except Exception:
                        pass
            except Exception as e:
                trace.append(f"{sym}:помилка відкриття")
                print(f"[FF-Engine] open error {sym}: {e}")

        # Prune attempt counters for coins that are no longer candidates.
        considered = set(cand)
        for k in list(self._engine_attempts):
            if k not in considered:
                self._engine_attempts.pop(k, None)

        # Persist the counters to DB if they changed this tick (survives restart).
        _persist_attempts_if_changed()

        # Diagnostics: expose per-coin skip reasons (parsed from the trace we
        # just built) so the ❤️ queue can show WHY each coin didn't open. The
        # engine ran through candidates, so the GLOBAL gate is clear.
        _skip = {}
        for _entry in trace:
            _sym, _sep, _reason = _entry.partition(':')
            if not _sep or _reason.startswith('✅'):
                continue   # opened (or malformed) → not a skip reason
            _skip[_sym] = _reason
        self._engine_skip = _skip
        self._engine_gate = ''

        print(f"[FF-Engine] {mode_lbl} · кнопки L={allow_long} S={allow_short} · "
              f"{len(cand)} канд · паливо-гейт · "
              + ' '.join(trace))

    def _queue_row(self, sym: str, info: Dict, now: float, smart: bool = False) -> Optional[Dict]:
        """Build ONE queue-table row for `sym`. Shared by Queue 1 and Queue 2 so
        both tables carry IDENTICAL columns. Returns None if the coin is already
        an open FF position (hidden from the waiting list).

        smart=True (🧭 Smart direction): the DISPLAYED direction is the live fuel
        (ММ) side, not the queued signal side — matching what will actually open.
        `smart_dir` flags rows whose fuel side differs from the entry signal."""
        if sym in self._fuel_managed:
            return None
        sig_d = info.get('dir')
        fuel_dir = (self._score_cache.get(sym) or {}).get('fuel_dir')
        if smart and fuel_dir in ('LONG', 'SHORT'):
            d = fuel_dir
            smart_flag = (fuel_dir != sig_d)
        else:
            d = sig_d
            smart_flag = False
        waited = now - info.get('added_at', now)
        return {
            'symbol': sym, 'dir': d,
            'signal_dir': sig_d,
            'smart_dir': smart_flag,
            'held_sec': int(waited),
            'waiting': True,
            'exhaustion': (self._score_cache.get(sym) or {}).get('exh'),
            'score': self._score_cache.get(sym),
            'mm': (self._score_cache.get(sym) or {}).get('fuel_dir'),
            'mm_str': self._fuel_str.get(sym),
            'mm_str_prev': self._fuel_str_prev.get(sym),
            'funding': False,
            'funding_rate': None,
            'funding_next_ms': None,
            'engine_attempts': self._engine_attempts.get(sym, 0),
            'engine_reason': self._engine_skip.get(sym) or None,
            'ctr': self._ctr_snapshot(sym),
        }

    def get_state(self) -> Dict:
        """Snapshot for the UI: settings + live timers + active tracking.
        NEW STRATEGY: shows only timers held >= duration_minutes (threshold).
        No auto-open. Position management (close) is optional via setting."""
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings['duration_minutes'] * 60
            funding_dur = float(settings.get('funding_duration_minutes', 0) or 0) * 60
            now = time.time()
            # The table = the waiting BASE (_pending). Show EVERY queued coin
            # (that isn't already an open position), REGARDLESS of the current
            # LONG/SHORT button state. Interception already gated entry by the
            # enabled button at ADD time — re-filtering here by the live buttons
            # made the WHOLE ❤️ queue vanish the moment the verdict flipped the
            # buttons to WAIT (both off), wiping coins that legitimately waited.
            # Buttons control OPENING (the engine), not visibility. The count
            # equals the rows shown.
            _smart = bool(settings.get('engine_smart_direction', False))
            # Queue 1 «❤️ Черга на вхід» (interception) rows.
            timers = []
            for sym, info in self._pending.items():
                row = self._queue_row(sym, info, now, _smart)
                if row:
                    timers.append(row)
            visible_pending = len(timers)
            all_timers = sorted(timers, key=lambda x: -x['held_sec'])
            # Queue 2 «⚡ CTR-зони» (scan) rows — identical columns.
            timers2 = []
            for sym, info in self._pending2.items():
                row = self._queue_row(sym, info, now, _smart)
                if row:
                    timers2.append(row)
            visible_pending2 = len(timers2)
            all_timers2 = sorted(timers2, key=lambda x: -x['held_sec'])
            bs = self._btc_state or {}
            # BTC START/STOP signal: progress counts up while the MAIN-WINDOW
            # verdict (compute_bias) holds LONG/SHORT, reaching START at
            # start_signal_minutes; STOP when the verdict is WAIT/None. The
            # banner is therefore 1:1 with 🎯 Smart Money Concepts (NOT liq-fuel).
            ssm = float(settings.get('start_signal_minutes', 5) or 5)
            period_sec = ssm * 60
            # SESSION direction (persists through WAIT). The timer keeps counting
            # through a pause, so START can still be reached while paused; the
            # `paused` flag lets the banner show "напрямок · ⏸ пауза".
            b_dir = self._btc_verdict_dir
            if b_dir in ('LONG', 'SHORT') and self._btc_verdict_since:
                b_held = now - self._btc_verdict_since
            else:
                b_dir = None
                b_held = 0
            has_btc = b_dir in ('LONG', 'SHORT')
            if not has_btc:
                btc_status, btc_prog = 'STOP', 0
            elif b_held >= period_sec:
                btc_status, btc_prog = 'START', 100
            else:
                btc_status = 'COUNTING'
                btc_prog = round(b_held / period_sec * 100, 1) if period_sec else 0
            btc_start = {
                'status': btc_status,
                'progress': btc_prog,
                'held_sec': int(b_held),
                'period_sec': int(period_sec),
                'dir': b_dir,
                'paused': bool(self._btc_paused and has_btc),
                # BTC fuel strength 0..100 — fills the banner bar (never empty).
                'strength': int(self._btc_fuel_strength or 0),
            }
            # 💰 Funding table — only coins currently holding fuel. Carries the
            # live funding rate + next-funding time + trend (prev_rate) + the
            # entry threshold so the UI can draw the entry→−4% progress bar.
            anomalies = []
            for sym, a in self._anomalies.items():
                held = int(now - a.get('started_at', now))
                anomalies.append({
                    'symbol': sym,
                    'dir': a.get('dir'),
                    'holding': True,
                    'held_sec': held,
                    'start_price': a.get('start_price'),
                    'current_price': a.get('last_price'),
                    'funding': True,
                    'funding_rate': a.get('rate'),
                    'funding_prev_rate': a.get('prev_rate'),
                    'funding_next_ms': a.get('next_funding'),
                    'entry_threshold': a.get('entry_threshold'),
                    'vol24h': a.get('vol24h'),
                    # ММ (fuel) direction + strength for the funding table's ММ
                    # column (same widget as the queue / open positions).
                    'mm': (self._score_cache.get(sym) or {}).get('fuel_dir') or a.get('dir'),
                    'mm_str': self._fuel_str.get(sym),
                    'mm_str_prev': self._fuel_str_prev.get(sym),
                    'paused': bool(a.get('sess_paused')),
                })
            anomalies.sort(key=lambda x: -x['held_sec'])
        # Live main-button gates (for engine_mode='buttons' UI + working dir).
        try:
            _eg_long, _eg_short = self._entry_gates()
        except Exception:
            _eg_long, _eg_short = True, True
        return {
            'ok': True,
            'settings': settings,
            'running': bool(self._thread and self._thread.is_alive()),
            'last_tick_ts': self._last_tick_ts,
            'timers': all_timers,
            # Queue 2 «⚡ CTR-зони» rows (same shape as `timers`).
            'timers2': all_timers2,
            'pending2_visible': visible_pending2,
            'queue1_enabled': bool(settings.get('queue1_enabled', True)),
            'queue2_enabled': bool(settings.get('queue2_enabled', False)),
            'btc_start': btc_start,
            'anomalies': anomalies,
            'active_symbols': list(self._fuel_managed.keys()),
            'tracked_count': len(self._fuel_managed),
            'scan_list': [],   # retired (FF no longer scans the WATCHLIST)
            'pending_count': len(self._pending),
            'pending_visible': visible_pending,   # rows actually shown (= header)
            'scan_stats': dict(self._scan_stats),
            # GLOBAL engine gate reason (empty when the engine is actively
            # scanning candidates). Shown above the ❤️ queue so the operator
            # instantly sees why NOTHING is opening (двигун вимкнено / BTC пауза
            # / чекає START / кнопки L/S).
            'engine_gate': self._engine_gate or '',
            # HOW the engine works right now: 'off' | 'btc' (₿ START session) |
            # 'buttons' (main-banner LONG/SHORT). Plus the live button gates so
            # the UI can show the working direction in buttons mode.
            'engine_mode': getattr(self, '_engine_mode', 'off'),
            'allow_long': _eg_long,
            'allow_short': _eg_short,
        }

    def active_symbols(self) -> List[str]:
        """Symbols with an open fuel-managed position — used to draw the ❤ marker
        in the watchlist."""
        with self._lock:
            return list(self._fuel_managed.keys())

    def is_in_table(self, symbol: str, side: str) -> bool:
        """Return True if `symbol` with direction `side` is currently shown in
        the ❤️ Fuel Auto-Filter table (i.e. its fuel timer has held for at least
        the configured `duration_minutes` threshold). Used as a confirmation gate
        for trade opens: a trade is only allowed if its coin+direction shows
        sustained fuel here."""
        symbol = (symbol or '').upper().strip()
        side = (side or '').upper().strip()
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings.get('duration_minutes', 5) * 60
            now = time.time()
            t = self._timers.get(symbol)
            if not t:
                return False
            if t.get('dir') != side:
                return False
            held = now - t.get('since', now)
            return held >= duration_sec

    def get_exhaustion_map(self) -> Dict[str, float]:
        """Get exhaustion values for the UI merge — for EVERY tracked symbol, not
        just FF-managed ones. Base layer = the background score cache ('exh',
        computed for all open TM positions + both queues); the live FF-manage
        value (fresher) overrides it where present. Returns {symbol: exh_pct}."""
        with self._lock:
            out = {}
            for sym, sc in self._score_cache.items():
                e = sc.get('exh')
                if e is not None:
                    out[sym] = e
            for sym, track in self._fuel_managed.items():
                e = track.get('exhaustion')
                if e is not None:
                    out[sym] = e   # live manage value takes precedence
            return out

    def get_fuel_strength_map(self) -> Dict[str, Dict]:
        """Fuel STRENGTH (0..100) + previous-cycle value + direction, per symbol
        we track (read from the pre-computed score cache — CHEAP, no per-request
        compute). For the open-position tables' 'Паливо' column.
        Returns {symbol: {'now': int, 'prev': int|None, 'dir': 'LONG'|'SHORT'|None}}."""
        with self._lock:
            out = {}
            for sym, sc in self._score_cache.items():
                st = sc.get('fuel_strength')
                if st is not None:
                    out[sym] = {'now': st,
                                'prev': self._fuel_str_prev.get(sym),
                                'dir': sc.get('fuel_dir')}
            return out

    def delete_timer(self, symbol: str) -> bool:
        """Remove a coin from the FF table — the waiting base (_pending) and/or
        a running timer. Returns True if anything was removed."""
        symbol = symbol.upper()
        with self._lock:
            removed = False
            if _q_allowed(5):   # OP 5: manual delete ✕
                removed = (self._pending.pop(symbol, None) is not None)
            if symbol in self._timers:
                self._timers.pop(symbol)
                removed = True
            if removed:
                self._persist_state()
                print(f"[FuelFilter] removed from FF table: {symbol}")
            return removed

    def force_open_timer(self, symbol: str) -> Dict:
        """Manually trigger position open for a running timer, bypassing the
        duration requirement (progress % doesn't matter).

        Returns: {'ok': bool, 'reason': str, 'opened': bool}
        """
        symbol = symbol.upper()
        settings = self.get_settings()
        if not settings.get('enabled'):
            return {'ok': False, 'reason': 'Fuel Auto-Filter is disabled'}

        with self._lock:
            timer = self._timers.get(symbol)
            pend = self._pending.get(symbol) or self._pending2.get(symbol)

        # Check if already holding a position
        if symbol in self._fuel_managed:
            return {'ok': False, 'reason': f'Already holding position for {symbol}'}

        # Direction comes from the waiting base (signal side) or, if already
        # opened-and-timing, from the timer. Force-open ignores the ₿ START /
        # fuel gate — the operator explicitly wants in.
        side = (pend or {}).get('dir') or (timer or {}).get('dir')
        if side not in ('LONG', 'SHORT'):
            return {'ok': False, 'reason': f'{symbol} не у черзі FF'}

        # Use the SAME fuel helper the tick loop uses (smoothed, read-only). It
        # returns {dir, mark_price, status} and has its own market_data fallback
        # for mark_price, so even with neutral/faded fuel we still get a price.
        fuel = self._fuel_dir_smoothed(symbol)
        if not fuel or not fuel.get('mark_price'):
            # Last-ditch: try a direct price fetch so a liq-map gap doesn't
            # block a manual open the user explicitly asked for.
            price = None
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    ticker = md.get_ticker(symbol)
                    price = ticker.get('last') if ticker else None
            except Exception:
                price = None
            if not price or price <= 0:
                return {'ok': False, 'reason': f'Не вдалося отримати ціну для {symbol}'}
            fuel = {'dir': (fuel.get('dir') if fuel else 0.0),
                    'mark_price': price,
                    'status': (fuel.get('status') if fuel else None)}

        try:
            opened = self._open(symbol, side, fuel, settings)
        except Exception as e:
            print(f"[FuelFilter] force_open_timer error for {symbol}: {e}")
            return {'ok': False, 'reason': f'Помилка: {str(e)}'}

        if opened:
            # Timer starts now (position lifetime); coin leaves the waiting base.
            with self._lock:
                self._timers[symbol] = {'dir': side, 'since': time.time(),
                                        'start_price': fuel.get('mark_price')}
                if _q_allowed(9):   # OP 9: manual force-open
                    self._pending.pop(symbol, None)
                    self._pending2.pop(symbol, None)
                self._persist_state()
            return {'ok': True, 'reason': f'Позицію {side} відкрито вручну', 'opened': True}
        else:
            # _open() returns False on its own gates (exhaustion too high,
            # WAIT verdict if skip_wait on, no entry price, qty below min, …).
            return {'ok': False, 'opened': False,
                    'reason': 'Відкриття відхилено (виснаженість/вердикт/розмір — див. лог)'}

    def clear_all_timers(self) -> int:
        """Clear the whole FF table — waiting base + any timers."""
        with self._lock:
            count = len(self._timers)
            if _q_allowed(6):   # OP 6: manual "Очистити всі"
                count += len(self._pending)
                self._pending.clear()
            self._timers.clear()
            self._persist_state()
            print(f"[FuelFilter] Cleared {count} timers")
            return count

    def delete_timer2(self, symbol: str) -> bool:
        """Remove ONE coin from Queue 2 «⚡ CTR-зони» (user ✕)."""
        symbol = (symbol or '').upper()
        with self._lock:
            removed = self._pending2.pop(symbol, None) is not None
            if removed:
                self._persist_state()
        return removed

    def clear_all_timers2(self) -> int:
        """Clear Queue 2 «⚡ CTR-зони» entirely (does NOT touch Queue 1)."""
        with self._lock:
            n = len(self._pending2)
            self._pending2 = {}
            self._persist_state()
        return n

    def delete_anomaly(self, symbol: str) -> bool:
        """Remove ONE coin from the anomalies table (user action)."""
        symbol = (symbol or '').upper()
        with self._lock:
            if symbol in self._anomalies:
                self._anomalies.pop(symbol, None)
                self._persist_state()
                print(f"[FuelFilter] Anomaly deleted: {symbol}")
                return True
            return False

    def clear_anomalies(self) -> int:
        """Clear the whole anomalies table. Returns count removed."""
        with self._lock:
            count = len(self._anomalies)
            self._anomalies.clear()
            self._persist_state()
            print(f"[FuelFilter] Cleared {count} anomalies")
            return count


_instance: Optional[FuelFilterDaemon] = None


def init_fuel_filter(db, get_trade_manager, get_watchlist) -> FuelFilterDaemon:
    """Create the singleton and auto-start the loop if the persisted toggle
    is ON (so it survives restarts)."""
    global _instance
    if _instance is None:
        _instance = FuelFilterDaemon(db, get_trade_manager, get_watchlist)
        if _instance.is_enabled():
            _instance.start()
            print("[FuelFilter] restored ON state from DB — loop running")
    return _instance


def get_fuel_filter() -> Optional[FuelFilterDaemon]:
    return _instance
